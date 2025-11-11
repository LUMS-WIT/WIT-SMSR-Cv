import os, torch, yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import SREncDataset, plot_predictions, plot_difference
from transformer import Transformer, TransformerSkip, GeoTransformerSR
from model import SRCNN_Shuffle


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

    # === Model ===
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

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            coarse, fine, center_lon, center_lat, date_str, *_ = batch
            coarse = coarse.to(device)
            pred = model(coarse, center_lat, center_lon, date_str) if geomodel else model(coarse)
            pred = pred.clamp(0.0, 1.0)

            plot_predictions(coarse.cpu(), fine.cpu(), pred.cpu(),
                             coarse_res=cfg["data"]["coarse_res"],
                             fine_res=cfg["data"]["fine_res"],
                             title=f"Prediction Sample #{idx}")

            plot_difference(fine.cpu(), pred.cpu(),
                            fine_res=cfg["data"]["fine_res"],
                            title=f"RMSE Sample #{idx}")

            if idx >= 1:  # show 2 samples max
                break


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)
