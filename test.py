import os, torch, yaml, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compare_images, mean_r, LossLogger, SREncDataset
from model import SRCNN_Shuffle
from transformer import Transformer, TransformerSkip, GeoTransformerSR
import matplotlib.pyplot as plt


def run(cfg, device):
    dataset = SREncDataset(
        coarse_dir=cfg["paths"]["coarse_dir"],
        fine_dir=cfg["paths"]["fine_dir"],
        coarse_res=cfg["data"]["coarse_res"],
        fine_res=cfg["data"]["fine_res"],
        scale_factor=cfg["data"]["scale_factor"],
        SMAP=cfg["data"]["smap"],
        upsample=cfg["data"]["upsample"]
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    geomodel = cfg["model"]["geomodel"]
    arch = cfg["model"]["arch"]
    scale_factor = cfg["data"]["scale_factor"]

    # === Model load ===
    if arch == "GeoTransformerSR":
        model = GeoTransformerSR(scale_factor=scale_factor).to(device)
    elif arch == "TransformerSkip":
        model = TransformerSkip(scale_factor=scale_factor).to(device)
    elif arch == "Transformer":
        model = Transformer(scale_factor=scale_factor).to(device)
    else:
        model = SRCNN_Shuffle(scale_factor=scale_factor).to(device)

    model.load_state_dict(torch.load(cfg["paths"]["model_path"], map_location=device, weights_only=True))
    model.eval()

    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc="Testing")):
            coarse, fine, center_lon, center_lat, date_str, *_ = batch
            coarse, fine = coarse.to(device), fine.to(device)
            pred = model(coarse, center_lat, center_lon, date_str) if geomodel else model(coarse)
            pred = pred.clamp(0.0, 1.0)
            metrics = compare_images(pred[0,0].cpu().numpy(), fine[0,0].cpu().numpy())
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    print("\nScores on Test Set:")
    for k, vals in all_metrics.items():
        mean_val = mean_r(vals) if k == "PearsonR" else np.mean(vals)
        print(f"{k}: {mean_val:.4f}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)
