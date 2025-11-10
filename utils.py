import numpy as np
import math
import json
import os
import glob
import re
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import generic_filter
from scipy.stats import pearsonr
from pyproj import Transformer

########################################################################################
# Data loader operations
########################################################################################

class SREncDataset(Dataset):
    def __init__(self, coarse_dir, fine_dir, coarse_res = "9km", fine_res="1km", scale_factor=9, 
                 upsample= True, legacy=False, SMAP=True, transform=None):
        self.coarse_dir = coarse_dir
        self.fine_dir = fine_dir
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.upsample = upsample
        self.scale_factor = scale_factor

        self.transform = transform
        self.legacy = legacy
        self.SMAP = SMAP

        if not SMAP:
            # Default logic
            self.coarse_files = sorted(glob.glob(os.path.join(coarse_dir, '*.tif')))
            self.fine_files = sorted([
                os.path.join(fine_dir, os.path.basename(f).replace(coarse_res, fine_res))
                for f in self.coarse_files
            ])

            # Filter out unmatched files
            self.coarse_files, self.fine_files = zip(*[
                (c, f) for c, f in zip(self.coarse_files, self.fine_files) if os.path.exists(f)
            ])

        else:
            # SMAP logic: match by YYYYMMDD_id
            coarse_files_all = sorted(glob.glob(os.path.join(coarse_dir, '*.tif')))
            fine_files_dict = {
                re.search(r'(\d{8}_\d+)', os.path.basename(f)).group(1): f
                for f in glob.glob(os.path.join(fine_dir, '*.tif'))
                if re.search(r'(\d{8}_\d+)', os.path.basename(f))
            }

            matched_pairs = []
            for cf in coarse_files_all:
                m = re.search(r'(\d{8}_\d+)', os.path.basename(cf))
                if m:
                    key = m.group(1)
                    if key in fine_files_dict:
                        matched_pairs.append((cf, fine_files_dict[key]))

            if not matched_pairs:
                raise RuntimeError("No matching coarse/fine SMAP file pairs found.")

            self.coarse_files, self.fine_files = zip(*matched_pairs)

    def __len__(self):
        return len(self.coarse_files)

    def upsample_coarse(self, coarse_tensor, scale_factor, mode='bilinear'):
        """
        Upsample a coarse-resolution tensor by a fixed scale factor.

        Args:
            coarse_tensor (Tensor): shape (1, H, W)
            scale_factor (int): Upscaling factor (e.g., 3 for 9km → 3km)
            mode (str): Interpolation mode ('bilinear', 'bicubic', etc.)

        Returns:
            upsampled_tensor (Tensor): shape (1, H * scale_factor, W * scale_factor)
        """
        up = F.interpolate(
            coarse_tensor.unsqueeze(0),  # [1, 1, H, W]
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False
        )
        return up.squeeze(0)  # return [1, H', W']


    def parse_metadata(self, filename):
        basename = os.path.basename(filename)
        # Example: SMAP-HB_3km_daily_mean_20150331_0.tif
        parts = basename.replace(".tif", "").split("_")

        platform = parts[0]                           # "SMAP_HB"
        res = parts[1]                                # "3km"
        date = parts[4]                               # "20150331"
        block_id = parts[5] if len(parts) > 5 else "0"

        return {
            "platform": platform,
            "res": res,
            "date": date,
            "id": block_id,
            "filename": basename
        }
   
    def fill_nan(self, arr, size=3):
        """
        Fill NaNs in the array using local mean filtering.
        """
        def mean_filter(window):
            valid = window[~np.isnan(window)]
            return valid.mean() if valid.size > 0 else np.nan
        return generic_filter(arr, mean_filter, size=size, mode='mirror')

    def __getitem__(self, idx):
        coarse_path = self.coarse_files[idx]
        fine_path = self.fine_files[idx]

        with rasterio.open(coarse_path) as csrc:
            coarse = csrc.read(1).astype(np.float32)
            transform = csrc.transform
            width = csrc.width          # (pixel_width: 8)
            height = csrc.height
            crs = csrc.crs
            res = csrc.res  

            # Compute center coordinates (pixel-wise)
            center_x = transform * (width // 2, height // 2)
            center_x = torch.tensor(center_x, dtype=torch.float32)
            try:
                # Convert to lat/lon if CRS is not EPSG:4326
                if crs and crs.to_epsg() != 4326:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    center_lon, center_lat = transformer.transform(center_x[0], center_x[1])
                else:
                    center_lon, center_lat = center_x
            except Exception:
                center_lon, center_lat = None, None

        with rasterio.open(fine_path) as fsrc:
            fine = fsrc.read(1).astype(np.float32)

        # Normalize (optional)
        coarse = (coarse - np.nanmin(coarse)) / (np.nanmax(coarse) - np.nanmin(coarse) + 1e-6)
        fine = (fine - np.nanmin(fine)) / (np.nanmax(fine) - np.nanmin(fine) + 1e-6)

        # Fill NaNs
        if np.isnan(fine).any():
            fine = self.fill_nan(fine)
            if np.isnan(fine).any():
                fine = self.fill_nan(fine)

        if np.isnan(coarse).any():
            coarse = self.fill_nan(coarse)

        # Add channel dimension
        coarse = torch.from_numpy(coarse).unsqueeze(0)
        fine = torch.from_numpy(fine).unsqueeze(0)

        if self.upsample:
            coarse = self.upsample_coarse(coarse, self.scale_factor)

        if self.transform:
            coarse, fine = self.transform((coarse, fine))

        if self.legacy:
            return coarse, fine
        else:
            meta = self.parse_metadata(coarse_path)
            date_str = meta['date']            # e.g. "20150331"
            date = torch.tensor(int(date_str), dtype=torch.int32) # YYYYMMDD → int
            # date  = datetime.strptime(date_str, "%Y%m%d")

            # Encode path (as UTF-8 bytes → uint8 tensor)
            path_bytes = coarse_path.encode("utf-8")
            coarse_path = torch.tensor(list(path_bytes), dtype=torch.uint8)
            # coarse_path = meta['filename']

            return coarse, fine, center_lon, center_lat, date, coarse_path
    
    def get_metadata(self, idx):

        """Return only metadata for the given index."""
        coarse_path = self.coarse_files[idx]
        fine_path = self.fine_files[idx]

        metadata_coarse = self.parse_metadata(coarse_path)
        metadata_fine = self.parse_metadata(fine_path)

        return metadata_coarse, metadata_fine


def plot_difference(fine, pred, idx=0, fine_res='', cmap='bwr', title=None, show_colorbar=True):
    """
    Plot the difference map between predicted and ground truth, with ubrmse in title.

    Args:
        pred (Tensor/ndarray): predicted, [B,1,H,W], [1,H,W], or [H,W]
        gt (Tensor/ndarray): ground truth, [B,1,H,W], [1,H,W], or [H,W]
        idx (int): batch index
        fine_res (str): resolution string for title
        cmap (str): colormap, default 'bwr' (blue-white-red)
        title (str): Optional user-supplied title
        show_colorbar (bool): whether to show colorbar
    """

    f_img = fine[idx, 0].numpy()
    p_img = pred[idx, 0].numpy()

    rmse_map = np.sqrt((p_img - f_img) ** 2)
    scalar_rmse = np.sqrt(np.mean((p_img - f_img) ** 2))
    u_rmse = ubrmse(f_img, p_img)

    full_title = f"RMSE Map (per-pixel) [RMSE={scalar_rmse:.4f}, ubRMSE={u_rmse:.4f}]"
    if fine_res:
        full_title += f" | Res: {fine_res}"
    if title:
        full_title = title + " | " + full_title

    plt.figure(figsize=(6, 5))
    im = plt.imshow(rmse_map, cmap=cmap)
    plt.title(full_title)
    plt.axis("off")
    if show_colorbar:
        plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.show()



def plot_sample(coarse, fine, idx=0, title=None, cmap='viridis'):
    """
    Plot coarse and fine resolution tensors side by side.
    
    Args:
        coarse (Tensor): shape [B, 1, H, W]
        fine (Tensor): shape [B, 1, H, W]
        idx (int): Index in the batch to visualize
        title (str): Optional plot title
        cmap (str): Matplotlib colormap
    """
    c_img = coarse[idx, 0].numpy()
    f_img = fine[idx, 0].numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(c_img, cmap=cmap)
    plt.title("Coarse")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(f_img, cmap=cmap)
    plt.title("Fine")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_predictions(coarse, fine, pred, idx=0, coarse_res='9km', 
                     fine_res='1km', title='Model Prediction', cmap='viridis'):
    """
    Plot coarse, groundtruth, and predicted images side by side.

    Accepts [B, 1, H, W], [1, H, W], or [H, W] arrays/tensors.
    """
    c_img = coarse[idx, 0].numpy()
    f_img = fine[idx, 0].numpy()
    p_img = pred[idx, 0].numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(c_img, cmap=cmap)
    plt.title(f"Coarse\n({coarse_res})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(f_img, cmap=cmap)
    plt.title(f"Ground Truth\n({fine_res})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(p_img, cmap=cmap)
    plt.title(f"Predicted\n({fine_res})")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

########################################################################################
## Metrics
########################################################################################

def psnr(target, ref):
    target_data = target.astype(np.float32)
    ref_data = ref.astype(np.float32)

    mse = np.mean((ref_data - target_data) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(1.0 / math.sqrt(mse))  # assuming values in [0, 1]

def mse_metric(target, ref):
    return np.mean((ref.astype('float32') - target.astype('float32')) ** 2)

def bias(target, ref):
    return np.mean(target) - np.mean(ref)

def ubrmse(target, ref):
    x = target - np.mean(target)
    y = ref - np.mean(ref)
    return np.sqrt(np.mean((x - y) ** 2))

def pearson_corr(target, ref):
    """
    Computes the Pearson correlation coefficient between target and ref.
    Returns just the correlation value (not the p-value).
    """
    return pearsonr(target.flatten(), ref.flatten())[0]


def compare_images(target, ref):
    """
    Args:
        target: NumPy array [H, W]
        ref: NumPy array [H, W]
    Returns:
        dict with PSNR, MSE, Bias, ubRMSE, Pearson Correlation
    """
    return {
        "PSNR": peak_signal_noise_ratio(target, ref, data_range=1),
        "MSE": mse_metric(target, ref),
        "Bias": bias(target, ref),
        "ubRMSE": ubrmse(target, ref),
        "PearsonR": pearson_corr(target, ref)
    }

def fisher_z(r):
    r = np.array(r)
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def inverse_fisher_z(z):
    return np.tanh(z)

def mean_r(correlations):
    z_values = fisher_z(correlations)
    mean_z = np.nanmean(z_values)
    return inverse_fisher_z(mean_z)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# logger.py

class LossLogger:
    def __init__(self, log_path="training_log.json", overwrite=False):
        self.log_path = log_path
        self.logs = {
            "train_loss": [],
            "train_psnr": [],
            "val_loss": [],
            "val_psnr": []
        }
        if not overwrite and os.path.exists(log_path):
            with open(log_path, "r") as f:
                self.logs = json.load(f)
        else:
            self._save()

    def log(self, train_loss, train_psnr, val_loss, val_psnr):
        self.logs["train_loss"].append(train_loss)
        self.logs["train_psnr"].append(train_psnr)
        self.logs["val_loss"].append(val_loss)
        self.logs["val_psnr"].append(val_psnr)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self):
        return self.logs

