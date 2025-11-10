import torch, os, time
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
from utils import LossLogger, SREncDataset
from model import SRCNN_Shuffle
from transformer import Transformer, TransformerSkip, GeoTransformerSR
import torch.nn.functional as F

# === Config params ===
COARSE_DIR = "training/train/9km"        # "training/central_valley/SMAP-HB/train/9km"
FINE_DIR = "training/train/1km"
UPSAMPLE = False # True for SRCNN or basleline test
COARSE_RES = "9km"
FINE_RES = "1km"
SAVE_DIR = "checkpoints"
SMAP= True
SCALE_FACTOR = 9
BATCH_SIZE = 8
EPOCHS = 10
VAL_SPLIT = 0.2
LR = 1e-3

GEOMODEL = True

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load full dataset ===
full_dataset = SREncDataset(
    coarse_dir=COARSE_DIR,
    fine_dir=FINE_DIR,
    coarse_res=COARSE_RES,
    fine_res=FINE_RES,
    scale_factor=SCALE_FACTOR,
    SMAP=SMAP,
    upsample=UPSAMPLE
)

# === Split train/val ===
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# === PSNR utility ===
def psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def batch_psnr(pred, target, max_val=1.0):
    # Flatten to (N, -1) for batchwise MSE
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # assumes shape (N, C, H, W)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr

def ubrmse(pred: torch.Tensor, gt: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Unbiased RMSE per sample: removes each sample’s mean before computing RMSE,
    then averages over the batch.
    """
    mp = pred.mean(dim=[1,2,3], keepdim=True)
    mg =   gt.mean(dim=[1,2,3], keepdim=True)
    pu, gu = pred - mp, gt - mg
    return torch.sqrt(((pu - gu)**2).mean(dim=[1,2,3]) + eps).mean()

def sr_loss(pred: torch.Tensor, gt: torch.Tensor,
            alpha=0, beta=0) -> torch.Tensor:
    """
    Combined SR loss:
      L = ubRMSE + alpha·MSE + beta·(1-SSIM)
      Note: Perfoming best on MSE only
    """
    loss_ub = ubrmse(pred, gt)
    loss_m  = F.mse_loss(pred, gt)
    return loss_m + alpha * loss_ub + beta 

# ─── Training & Evaluation Routines ────────────────────────────────────────
def model_forward(model, batch):
    """
    Forward pass for both legacy and new models.
    batch: tuple returned by DataLoader
    Returns: (output, fine)
    """
    if len(batch) == 2:
        # Legacy: (coarse, fine)
        coarse, fine = batch
        out = model(coarse)
        return out, fine
    elif len(batch) > 2:
        # New: (coarse, fine, center_lon, center_lat, date_str, coarse_path)
        if GEOMODEL:
            coarse, fine, center_lon, center_lat, date_str, coarse_path = batch
            out = model(coarse, center_lat, center_lon, date_str)
        else:
            coarse, fine, *_ = batch  # Ignore extras if present
            out = model(coarse)
        return out, fine
    else:
        raise ValueError(f"Unexpected batch structure: {len(batch)} elements")

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        # Move all tensors to device
        batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)

        optimizer.zero_grad()
        out, fine = model_forward(model, batch)
        loss = sr_loss(out, fine)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Move all tensors to device
            batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
            pred, fine = model_forward(model, batch)
            pred = pred.clamp(0.0, 1.0)

            total_loss += F.mse_loss(pred, fine).item()
            psnrs = batch_psnr(pred, fine)
            total_psnr += psnrs.sum().item()
            total_samples += pred.size(0)

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / total_samples
    return avg_loss, avg_psnr




# === Model, optimizer, loss ===
# model = SRCNN_Shuffle(scale_factor=SCALE_FACTOR).to(device)
# model = Transformer(scale_factor=SCALE_FACTOR).to(device)
# model = TransformerSkip(scale_factor=SCALE_FACTOR).to(device)
model = GeoTransformerSR(scale_factor=SCALE_FACTOR).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


# === Best Model Tracking ===
best_psnr = 10.0
best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
log_path = os.path.join(SAVE_DIR, "training_log.json")
loss_logger = LossLogger(log_path=log_path, overwrite=True)

start_time = time.time()
# === Training loop ===
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    train_psnr = evaluate(model, train_loader, device)[1]  # Only get PSNR

    val_loss, val_psnr = evaluate(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} | "
          f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}")

    loss_logger.log(train_loss, train_psnr, val_loss, val_psnr)

    # === Save best model ===
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model (PSNR: {best_psnr:.2f})")

end_time = time.time()
elasped_time = end_time - start_time
hours = elasped_time // 3600
minutes = (elasped_time % 3600) // 60

print(f"\nTraining complete. Best model saved at: {best_model_path} (PSNR: {best_psnr:.2f})")
print(f"Total training time: {int(hours)} hours {int(minutes)} minutes")
