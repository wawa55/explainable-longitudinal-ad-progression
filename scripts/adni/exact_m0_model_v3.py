import torch
import torch.nn as nn


class ExactResidualBlock3D_V3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        else:
            self.skip = nn.Identity()

        self.out_act = nn.GELU()

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        return self.out_act(x)


class ExactMRIEncoder3D_V3(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.GELU(),
        )
        self.stage1 = ExactResidualBlock3D_V3(16, 32, stride=2, dropout=dropout)
        self.stage2 = ExactResidualBlock3D_V3(32, 64, stride=2, dropout=dropout)
        self.stage3 = ExactResidualBlock3D_V3(64, 128, stride=2, dropout=dropout)
        self.stage4 = ExactResidualBlock3D_V3(128, 192, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(192, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.projection(x)


class ExactContinuousTimeEncoding_V3(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, years_from_first, years_before_landmark):
        features = torch.stack([years_from_first, years_before_landmark], dim=-1)
        return self.network(features)


class ExactM0LongitudinalSurvivalModel_V3(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        n_heads=8,
        n_transformer_layers=3,
        ff_dim=512,
        dropout=0.1,
        n_intervals=5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mri_encoder = ExactMRIEncoder3D_V3(embedding_dim, dropout)
        self.time_encoder = ExactContinuousTimeEncoding_V3(embedding_dim, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.survival_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_intervals),
        )

    def forward(self, images, visit_mask, years_from_first, years_before_landmark):
        if images.ndim != 6:
            raise ValueError("Expected images with shape [B,T,C,D,H,W].")

        B, T, C, D, H, W = images.shape
        flat_images = images.reshape(B * T, C, D, H, W)
        flat_mask = visit_mask.reshape(B * T)

        embeddings = torch.zeros(
            (B * T, self.embedding_dim),
            dtype=images.dtype,
            device=images.device,
        )

        if flat_mask.any():
            embeddings[flat_mask] = self.mri_encoder(flat_images[flat_mask])

        mri_embeddings = embeddings.reshape(B, T, self.embedding_dim)
        time_embeddings = self.time_encoder(years_from_first, years_before_landmark)
        x = mri_embeddings + time_embeddings

        mask_float = visit_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask_float
        x = self.transformer(x, src_key_padding_mask=(~visit_mask))
        x = self.final_norm(x)

        pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        return self.survival_head(pooled)
