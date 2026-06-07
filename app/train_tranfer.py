"""
ADAS Traffic Light Classifier — Transfer Learning Trainer
==========================================================
Fine-tunes a MobileNetV2 backbone on cropped traffic-light images and
exports the result as an ONNX model that plugs directly into the Rust
AdasBrain engine.

Dataset layout (folder name → class index):
    raw_lights/
        0_red/       → 0
        1_yellow/    → 1
        2_green/     → 2
        3_off/       → 3

ONNX contract (must match lib.rs AdasBrain):
    Input:  "input"   float32 [batch, 3, 64, 32]   (CHW, RGB, normalised)
    Output: "output"  float32 [batch, 4]            (logits)
"""

import os
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR      = r"C:\Users\CREWMOBILE\Videos\raw_lights"
ONNX_OUT      = r"..\models\traffic_lights_transfer.onnx"
IMG_HEIGHT    = 64
IMG_WIDTH     = 32
NUM_CLASSES   = 4
BATCH_SIZE    = 32
NUM_EPOCHS    = 40
LEARNING_RATE = 1e-3       # For the new head
LR_BACKBONE   = 1e-5      # Unfrozen backbone layers (much smaller)
VAL_SPLIT     = 0.2        # 20% held out for validation
SEED          = 42

CLASS_NAMES = ["RED", "YELLOW", "GREEN", "OFF"]


# ==========================================
# DATA AUGMENTATION
# ==========================================
train_transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.RandomHorizontalFlip(p=0.3),
    T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.85, 1.15)),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])

val_transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ==========================================
# HELPER CLASSES / FUNCTIONS
# ==========================================
class SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a Subset so train/val can have different transforms."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def split_dataset(full_dataset, val_ratio, seed):
    """Stratified-ish split that respects class balance."""
    n = len(full_dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * val_ratio)
    return indices[split:], indices[:split]   # train, val


def run_epoch(model, loader, criterion, device, optimizer=None, is_train=True):
    """Run one training or validation epoch."""
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


# ==========================================
# MAIN
# ==========================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------
    print(f"\n[INFO] Loading dataset from: {DATA_DIR}")
    full_dataset = ImageFolder(DATA_DIR)
    print(f"[INFO] Found {len(full_dataset)} images across {len(full_dataset.classes)} classes")
    print(f"[INFO] Class mapping: {full_dataset.class_to_idx}")

    train_idx, val_idx = split_dataset(full_dataset, VAL_SPLIT, SEED)
    print(f"[INFO] Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    train_set = SubsetWithTransform(full_dataset, train_idx, train_transform)
    val_set   = SubsetWithTransform(full_dataset, val_idx,   val_transform)

    # ------------------------------------------
    # 2. WEIGHTED SAMPLER (handles class imbalance)
    # ------------------------------------------
    train_labels = [full_dataset.targets[i] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.clip(class_counts, 1.0, None)

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )

    print(f"[INFO] Class distribution (train): {dict(zip(CLASS_NAMES, class_counts.astype(int)))}")
    print(f"[INFO] Sampling weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    # num_workers=0 to avoid Windows multiprocessing issues
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ------------------------------------------
    # 3. BUILD MODEL
    # ------------------------------------------
    print("\n[INFO] Building MobileNetV2 transfer-learning model...")
    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze early layers, unfreeze last 4 inverted-residual blocks
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.features[-4:].parameters():
        param.requires_grad = True

    # Replace classifier head
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(backbone.last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, NUM_CLASSES),
    )

    model = backbone.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable:,} / {total_params:,}")

    # ------------------------------------------
    # 4. OPTIMIZER & SCHEDULER
    # ------------------------------------------
    param_groups = [
        {"params": model.features[-4:].parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(),     "lr": LEARNING_RATE},
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Weighted cross-entropy to penalise misclassifying rare classes
    ce_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weights)

    # ------------------------------------------
    # 5. TRAINING LOOP
    # ------------------------------------------
    print(f"\n{'='*60}")
    print(f"  TRAINING — {NUM_EPOCHS} epochs, batch={BATCH_SIZE}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    PATIENCE = 10

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device,            is_train=False)

        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[-1]['lr']
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:5.1f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:5.1f}%  |  "
              f"LR: {lr_now:.2e}  |  {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ New best val accuracy: {val_acc*100:.1f}%")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\n[INFO] Best validation accuracy: {best_val_acc*100:.1f}%")

    # ------------------------------------------
    # 6. CONFUSION MATRIX
    # ------------------------------------------
    print(f"\n{'='*60}")
    print("  VALIDATION CONFUSION MATRIX")
    print(f"{'='*60}")

    model.load_state_dict(best_model_state)
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            for true, pred in zip(labels.numpy(), preds):
                confusion[true][pred] += 1

    header = "True \\ Pred"
    print(f"\n  {header:>12s}  " + "  ".join(f"{c:>8s}" for c in CLASS_NAMES))
    print("  " + "-" * (14 + 10 * NUM_CLASSES))
    for i, row in enumerate(confusion):
        row_str = "  ".join(f"{v:8d}" for v in row)
        total_row = row.sum()
        acc = row[i] / total_row * 100 if total_row > 0 else 0
        print(f"  {CLASS_NAMES[i]:>12s}  {row_str}  ({acc:.0f}%)")

    overall = confusion.trace() / confusion.sum() * 100 if confusion.sum() > 0 else 0
    print(f"\n  Overall accuracy: {overall:.1f}%")

    # ------------------------------------------
    # 7. ONNX EXPORT
    # ------------------------------------------
    print(f"\n{'='*60}")
    print("  EXPORTING TO ONNX")
    print(f"{'='*60}")

    model.eval()
    dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH, device=device)

    onnx_path = os.path.abspath(ONNX_OUT)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[INFO] Exported to: {onnx_path}")
    print(f"[INFO] Model size:  {file_size:.2f} MB")

    # ------------------------------------------
    # 8. SANITY CHECK
    # ------------------------------------------
    print("\n[INFO] Running ONNX sanity check...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)
    result = ort_session.run(None, {"input": test_input})
    print(f"[INFO] ONNX output shape: {result[0].shape}  (expected: (1, {NUM_CLASSES}))")
    print(f"[INFO] Sample logits: {result[0][0]}")
    pred_class = CLASS_NAMES[result[0][0].argmax()]
    print(f"[INFO] Predicted class (random noise): {pred_class}")

    print(f"\n{'='*60}")
    print("  DONE — Model ready for deployment")
    print(f"  Reload AdasBrain in main.py to use the updated weights.")
    print(f"{'='*60}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()