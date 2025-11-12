import os, time, torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import LossLogger, SREncDataset
from model import SRCNN_Shuffle
from transformer import Transformer, TransformerSkip, GeoTransformerSR
import yaml


# === Core Functions ===
def psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))

def batch_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def ubrmse(pred, gt, eps=1e-6):
    mp, mg = pred.mean(dim=[1,2,3], keepdim=True), gt.mean(dim=[1,2,3], keepdim=True)
    return torch.sqrt(((pred - mp - (gt - mg))**2).mean(dim=[1,2,3]) + eps).mean()

def sr_loss(pred, gt, alpha=0, beta=0):
    loss_ub = ubrmse(pred, gt)
    loss_m  = F.mse_loss(pred, gt)
    return loss_m + alpha * loss_ub + beta


def model_forward(model, batch, geomodel):
    if len(batch) == 2:
        coarse, fine = batch
        out = model(coarse)
        return out, fine
    elif len(batch) > 2:
        coarse, fine, center_lon, center_lat, date_str, *_ = batch
        out = model(coarse, center_lat, center_lon, date_str) if geomodel else model(coarse)
        return out, fine
    else:
        raise ValueError(f"Unexpected batch structure: {len(batch)} elements")


def train_one_epoch(model, loader, optimizer, device, geomodel):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
        optimizer.zero_grad()
        out, fine = model_forward(model, batch, geomodel)
        loss = sr_loss(out, fine)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, geomodel):
    model.eval()
    total_loss, total_psnr, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
            pred, fine = model_forward(model, batch, geomodel)
            pred = pred.clamp(0.0, 1.0)
            total_loss += F.mse_loss(pred, fine).item()
            total_psnr += batch_psnr(pred, fine).sum().item()
            total_samples += pred.size(0)
    return total_loss / len(loader), total_psnr / total_samples


# === Main Entrypoint ===
def run(cfg, device):
    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)

    dataset = SREncDataset(
        coarse_dir=cfg["paths"]["coarse_dir"],
        fine_dir=cfg["paths"]["fine_dir"],
        coarse_res=cfg["data"]["coarse_res"],
        fine_res=cfg["data"]["fine_res"],
        scale_factor=cfg["data"]["scale_factor"],
        SMAP=cfg["data"]["smap"],
        upsample=cfg["data"]["upsample"],
    )

    val_split = cfg["training"]["val_split"]
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    geomodel = cfg["model"]["geomodel"]
    arch = cfg["model"]["arch"]
    scale_factor = cfg["data"]["scale_factor"]

    # === Model ===
    if arch == "GeoTransformerSR":
        model = GeoTransformerSR(scale_factor=scale_factor).to(device)
    elif arch == "TransformerSkip":
        model = TransformerSkip(scale_factor=scale_factor).to(device)
    elif arch == "Transformer":
        model = Transformer(scale_factor=scale_factor).to(device)
    else:
        model = SRCNN_Shuffle(scale_factor=scale_factor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    best_psnr, best_model_path = 10.0, os.path.join(cfg["paths"]["save_dir"], "best_model.pth")
    
    logging = cfg["training"]["log"]
    if logging:
        logger = LossLogger(os.path.join(cfg["paths"]["save_dir"], "training_log.json"), overwrite=True)

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, geomodel)
        train_psnr = evaluate(model, train_loader, device, geomodel)[1]
        val_loss, val_psnr = evaluate(model, val_loader, device, geomodel)

        print(f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} | "
              f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}")

        if logging:
            logger.log(train_loss, train_psnr, val_loss, val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (PSNR: {best_psnr:.2f})")

    print(f"\nTraining done. Best model: {best_model_path} (PSNR: {best_psnr:.2f})")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)
