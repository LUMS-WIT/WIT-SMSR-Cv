import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from datetime import datetime

class Transformer(nn.Module):
    
    def __init__(self, input_size=8, scale_factor=3, patch_size=4, dim=128, heads=4, mlp_dim=256, depth=4, dropout=0.1):
        super().__init__()

        self.scale = scale_factor
        self.input_size = input_size
        self.output_size = input_size * scale_factor
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.upsampled_patches = (self.output_size // patch_size) ** 2
        self.embed_dim = dim
        self.channels = 1

        patch_dim = self.channels * patch_size * patch_size # 1×4×4 = 16

        # Linear patch embedding : 16 → 128
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed_enc = nn.Parameter(torch.randn(1, self.num_patches, dim)) # [1, 4, 128]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Upsample tokens (dimension preserved: 128 → 128)
        self.upsample_tokens = nn.Linear(dim, dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.pos_embed_dec = nn.Parameter(torch.randn(1, self.upsampled_patches, dim)) # [1, 36, 128]

        # Output projection : Project decoder output back to pixel patches: 128 → 16
        self.output_proj = nn.Linear(dim, patch_dim)

    def forward(self, x):
        """
        x: [B, 1, H, W] where H=W=input_size, e.g., [8, 1, 8, 8]
        Output: [B, 1, H*scale, W*scale], e.g., [8, 1, 24, 24]
        """
        B, C, H, W = x.shape    # [8, 1, 8, 8]
        p = self.patch_size

        # 1. Patchify input: -> [B, num_patches, patch_dim]
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        # => [8, 4, 16]

        # 2. Linear embed + positional encodings → [B, num_patches, dim]
        x = self.patch_embed(x) + self.pos_embed_enc[:, :self.num_patches]  # [8, 4, 128]
        x = self.encoder(x)  # [B, num_patches, dim]
        # => [8, 4, 128]

        # 3. Upsample tokens : [B, 4, 128] → [B, 36, 128] via repeat
        x = self.upsample_tokens(x) # [8, 4, 128]
        x = x.repeat_interleave((self.upsampled_patches // self.num_patches), dim=1)  # Token expansion
        # => [8, 36, 128]

        # 4. Positional encoding for decoder input
        tgt = torch.zeros_like(x) + self.pos_embed_dec[:, :x.shape[1]]
        x = self.decoder(tgt, x)
        # => [8, 36, 128]        

        # 5. Project to pixel space
        x = self.output_proj(x)  # [B, up_patches, patch_dim] # [8, 36, 16]
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', 
                      h=self.output_size // p, w=self.output_size // p, p1=p, p2=p, c=C)

        # => [8, 1, 24, 24]
        return x


class TransformerSkip(nn.Module):

    def __init__(self, input_size=8, scale_factor=3, patch_size=4, dim=128, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()

        self.scale = scale_factor
        self.input_size = input_size
        self.output_size = input_size * scale_factor
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.upsampled_patches = (self.output_size // patch_size) ** 2
        self.embed_dim = dim
        self.channels = 1

        patch_dim = self.channels * patch_size * patch_size  # 1x4x4 = 16

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed_enc = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.pos_embed_dec = nn.Parameter(torch.randn(1, self.upsampled_patches, dim))

        # Encoder stages
        def make_encoder():
            layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
            return nn.TransformerEncoder(layer, num_layers=1)

        self.encoder1 = make_encoder()
        self.encoder2 = make_encoder()
        self.encoder3 = make_encoder()
        self.encoder4 = make_encoder()  # For upsampled tokens

        self.upsample_tokens = nn.Linear(dim, dim)

        # Decoder stages
        def make_decoder():
            layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
            return nn.TransformerDecoder(layer, num_layers=1)

        self.decoder3 = make_decoder()
        self.decoder2 = make_decoder()
        self.decoder1 = make_decoder()

        self.output_proj = nn.Linear(dim, patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # 1. Patchify
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)  # [B, 4, 16]

        # 2. Linear Embedding + Positional Encoding
        x = self.patch_embed(x) + self.pos_embed_enc[:, :self.num_patches]  # [B, 4, D]

        # 3. Encoder stages with skip connections
        x1 = self.encoder1(x)  # [B, 4, D]
        x2 = self.encoder2(x1)  # [B, 4, D]
        x3 = self.encoder3(x2)  # [B, 4, D]

        # 4. Token Expansion to match upsampled size
        x_up = self.upsample_tokens(x3)  # [B, 4, D]
        x_up = x_up.repeat_interleave(self.upsampled_patches // self.num_patches, dim=1)  # [B, 36, D]
        x_up = self.encoder4(x_up)  # [B, 36, D]

        # 5. Decoder stages with skip attention
        tgt = torch.zeros_like(x_up) + self.pos_embed_dec[:, :x_up.shape[1]]
        x_up = self.decoder3(tgt, x3)  # attends on deep skip
        x_up = self.decoder2(x_up, x2)  # attends on mid skip
        x_up = self.decoder1(x_up, x1)  # attends on shallow skip

        # 6. Output projection and unpatchify
        x_out = self.output_proj(x_up)  # [B, 36, 16]
        x_out = rearrange(x_out, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                          h=self.output_size // p, w=self.output_size // p, p1=p, p2=p, c=C)

        return x_out


################################################################################
## GeoTrnasformerSR
################################################################################

# ─── Helpers ────────────────────────────────────────────────────────────────

def posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature: int = 10000):
    """
    2D sin/cos positional encoding on an h x w grid, with frequencies scaled by GSD.
    Returns: Tensor of shape [h*w, dim]
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "dim must be multiple of 4"
    
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4 - 1)
    omega = (1.0 / (temperature ** (2 * omega / dim))) * gsd
    
    x = x.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)  # [h*w, dim]
    
    return pe


def normalize_metadata(lat, lon, date):
    """
    Turns (lat, lon, date) into an 8‑vector:
      [sin(lat), cos(lat),
       sin(lon), cos(lon),
       sin(day_of_year), cos(day_of_year),
       sin(month_of_year), cos(month_of_year)]
    """
    # 1) lat/lon → radians
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat_s, lat_c = math.sin(lat_r), math.cos(lat_r)
    lon_s, lon_c = math.sin(lon_r), math.cos(lon_r)

    dt = datetime.strptime(str(date.item()), "%Y%m%d")
    doy= dt.timetuple().tm_yday
    moy= dt.month


    # doy = date.timetuple().tm_yday
    angle_doy = 2 * math.pi * (doy / 365.0)
    doy_s, doy_c = math.sin(angle_doy), math.cos(angle_doy)

    # 3) month-of-year cycle (1–12)
    # moy = date.month - 1   # 0–11
    moy = moy - 1
    angle_moy = 2 * math.pi * (moy / 12.0)
    moy_s, moy_c = math.sin(angle_moy), math.cos(angle_moy)

    return [lat_s, lat_c, lon_s, lon_c, doy_s, doy_c, moy_s, moy_c]


class GeoTransformerSR(nn.Module):
    def __init__(
        self,
        input_size=8,
        scale_factor=3,
        patch_size=4,
        dim=128,
        heads=4,
        mlp_dim=256,
        depth=4,
        dropout=0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.scale = scale_factor
        self.output_size = input_size * scale_factor
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.upsampled_patches = (self.output_size // patch_size) ** 2
        self.embed_dim = dim
        self.channels = 1

        patch_dim = self.channels * patch_size * patch_size  # e.g., 1×4×4

        # ─ Patch embedding (pixel → token)
        self.patch_embed = nn.Linear(patch_dim, dim)

        # ─ Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # ─ Upsample tokens (dim stays same)
        self.upsample_tokens = nn.Linear(dim, dim)

        # ─ Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=depth)

        # ─ Output projection (token → pixel patch)
        self.output_proj = nn.Linear(dim, patch_dim)

    def forward(self, x, lats, lons, dates, gsd=1.0):
        """
        x: [B, 1, H, W]
        lats, lons: lists or tensors of length B
        dates: list of datetime.datetime of length B
        gsd: float (meters per pixel)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        L = self.num_patches

        # 1) Patchify & embed
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)   # → [B, L, patch_dim]
        x = self.patch_embed(x)                                                # → [B, L, dim]

        # 2) Clay‑style 2D sin/cos pos‑encoding + global metadata
        # 2a) 2D grid encoding
        grid_size = self.input_size // p
        pos_enc = posemb_sincos_2d_with_gsd(
            h=grid_size, w=grid_size, dim=self.embed_dim - 8, gsd=gsd
        ).to(x.device)                                                          # [L, dim-8]
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)                       # [B, L, dim-8]

        # 2b) global metadata
        meta = torch.tensor(
            [normalize_metadata(lats[i], lons[i], dates[i]) for i in range(B)],
            dtype=x.dtype, device=x.device
        )                                                                       # [B,8]
        meta = meta.unsqueeze(1).expand(B, L, -1)                               # → [B,L,8]

        # 2c) concat & add
        pos_meta = torch.cat((pos_enc, meta), dim=-1)                          # [B, L, dim]
        x = x + pos_meta

        # 3) Encode
        x = self.encoder(x)                                                     # [B, L, dim]

        # 4) Upsample tokens
        x = self.upsample_tokens(x)                                             # [B, L, dim]
        factor = self.upsampled_patches // L
        x = x.repeat_interleave(factor, dim=1)                                  # [B, up_patches, dim]

        # 5) Decoder positional+metadata (for upsampled grid)
        # 5a) new grid encoding
        grid_out = self.output_size // p
        pos_dec = posemb_sincos_2d_with_gsd(
            h=grid_out, w=grid_out, dim=self.embed_dim - 8, gsd=gsd
        ).to(x.device)                                                          # [up_patches, dim-8]
        pos_dec = pos_dec.unsqueeze(0).expand(B, -1, -1)                        # [B, up_patches, dim-8]

        # 5b) same global metadata
        meta_dec = meta[:, :1, :].expand(B, self.upsampled_patches, 8)          # [B, up_patches, 8]

        # 5c) concat
        tgt = torch.cat((pos_dec, meta_dec), dim=-1)                            # [B, up_patches, dim]

        # 6) Decode
        x = self.decoder(tgt, x)                                                # [B, up_patches, dim]

        # 7) Project back to pixels & rearrange
        x = self.output_proj(x)                                                 # [B, up_patches, patch_dim]
        x = rearrange(
            x,
            'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            h=grid_out, w=grid_out, p1=p, p2=p, c=C
        )                                                                       # [B, 1, H*scale, W*scale]
        return x
