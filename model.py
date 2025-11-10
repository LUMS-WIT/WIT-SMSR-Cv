import torch.nn as nn
import torch.nn.functional as F

class SRCNN_Shuffle(nn.Module):
    def __init__(self, scale_factor=3):
        super(SRCNN_Shuffle, self).__init__()

        # Downsample: 24x24 → 12x12
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=3 if scale_factor==9 else 2, padding=1),  # → [B, 64, 12, 12]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # → [B, 128, 12, 12]
            nn.ReLU(inplace=True)
        )

        if scale_factor == 3:
            # Prepare for PixelShuffle upsampling: output should have C = upscale^2 * channels
            self.upsample = nn.Sequential(
                nn.Conv2d(128, 4, kernel_size=3, padding=1),            # → [B, 4, 12, 12]
                nn.PixelShuffle(upscale_factor=2),                     # → [B, 1, 24, 24]
            )

        elif scale_factor == 9:
            # Upsample by 3: output C = 1*3*3=9
            self.upsample = nn.Sequential(
                nn.Conv2d(128, 9, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=3),  # Restore to input size
            )
        else:
            raise ValueError("Only scale_factor 3 or 9 is supported.")
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.upsample(x)
        return x