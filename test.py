import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compare_images, mean_r, SREncDataset
from model import SRCNN_Shuffle
from transformer import Transformer, TransformerSkip, GeoTransformerSR


def evaluate_model(model, loader, device, geomodel):
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}
    model.eval()

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc="Testing")):
            if len(batch) == 2:
                coarse, fine = batch
                center_lon = center_lat = date_str = None
            else:
                coarse, fine, center_lon, center_lat, date_str, *_ = batch

            coarse, fine = coarse.to(device), fine.to(device)
            pred = model(coarse, center_lat, center_lon, date_str) if geomodel else model(coarse)
            pred = pred.clamp(0.0, 1.0)

            pred_np = pred[0, 0].cpu().numpy()
            fine_np = fine[0, 0].cpu().numpy()
            metrics = compare_images(pred_np, fine_np)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    results = {k: mean_r(v) if k == "PearsonR" else np.mean(v) for k, v in all_metrics.items()}
    return results


def evaluate_baseline(cfg):
    dataset = SREncDataset(
        coarse_dir=cfg["paths"]["coarse_dir"],
        fine_dir=cfg["paths"]["fine_dir"],
        coarse_res=cfg["data"]["coarse_res"],
        fine_res=cfg["data"]["fine_res"],
        scale_factor=cfg["data"]["scale_factor"],
        SMAP=cfg["data"]["smap"],
        upsample=True,  # Baseline = upsample coarse → fine
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    for batch_id, batch in enumerate(tqdm(loader, desc="Baseline")):
        if len(batch) == 2:
            coarse, fine = batch
        else:
            coarse, fine, *_ = batch

        coarse_np = coarse[0, 0].cpu().numpy()
        fine_np = fine[0, 0].cpu().numpy()
        metrics = compare_images(np.clip(coarse_np, 0, 1), np.clip(fine_np, 0, 1))
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    results = {k: mean_r(v) if k == "PearsonR" else np.mean(v) for k, v in all_metrics.items()}
    return results


def run(cfg, device):
    dataset = SREncDataset(
        coarse_dir=cfg["paths"]["coarse_dir"],
        fine_dir=cfg["paths"]["fine_dir"],
        coarse_res=cfg["data"]["coarse_res"],
        fine_res=cfg["data"]["fine_res"],
        scale_factor=cfg["data"]["scale_factor"],
        SMAP=cfg["data"]["smap"],
        upsample=cfg["data"]["upsample"],
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    geomodel = cfg["model"]["geomodel"]
    arch = cfg["model"]["arch"]
    scale_factor = cfg["data"]["scale_factor"]

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

    # === Model evaluation ===
    model_scores = evaluate_model(model, loader, device, geomodel)
    print("\nScores on Test Set (Model):")
    for k, v in model_scores.items():
        print(f"{k}: {v:.4f}")

    # === Baseline evaluation ===
    baseline_scores = evaluate_baseline(cfg)
    print("\nBaseline on Test Set:")
    for k, v in baseline_scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)
