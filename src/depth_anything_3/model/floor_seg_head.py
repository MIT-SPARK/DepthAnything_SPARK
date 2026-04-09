import torch
import torch.nn as nn
import torch.nn.functional as F


class FloorSegHead(nn.Module):
    """
    Lightweight floor segmentation head for Depth Anything 3.

    Receives patch tokens from a chosen ViT layer (extracted externally via
    backbone.pretrained.get_intermediate_layers) and predicts a binary floor mask.

    Args:
        dim_in (int): Input token dimension.  2 * embed_dim because DA3's backbone
                      concatenates local and global attention streams.
                      Default 3072 (2 * 1536 for vitg / da3-giant).
        patch_size (int): Patch size used by the ViT backbone. Default 14.
    """

    def __init__(self, dim_in: int = 3072, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size

        self.norm = nn.LayerNorm(dim_in)

        self.decoder = nn.Sequential(
            # 1x1 bottleneck
            nn.Conv2d(dim_in, 256, kernel_size=1),
            nn.GELU(),
            # 3x3 refinement
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            # 2x upsample: ph -> 2ph
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            # 2x upsample: 2ph -> 4ph
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _apply(self, fn):
        # Intercept dtype conversions (e.g. model.to(bfloat16)) to keep this
        # module's parameters in float32 while allowing device moves.
        probe = fn(torch.empty(1, dtype=torch.float32))
        if probe.dtype == torch.float32:
            return super()._apply(fn)
        def _keep_float32(t):
            result = fn(t)
            return result.float() if t.dtype == torch.float32 else result
        return super()._apply(_keep_float32)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, S, N_patch, dim_in] patch tokens from a single ViT layer.
                    Extracted via backbone.pretrained.get_intermediate_layers(x, n=[layer_idx])[0][0][0].
            images: [B, S, 3, H, W] input images (used only to determine spatial output size).

        Returns:
            logits: [B, S, H, W] raw (pre-sigmoid) floor segmentation logits.
        """
        B, S, _, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size

        # Flatten batch and sequence dims: [B*S, N_patch, dim_in]
        x = tokens.reshape(B * S, -1, tokens.shape[-1])

        # Cast to float32; seg head weights always stay float32 (see _apply above).
        x = x.float()
        x = self.norm(x)

        # Reshape to spatial feature map: [B*S, dim_in, ph, pw]
        x = x.permute(0, 2, 1).reshape(B * S, tokens.shape[-1], ph, pw)

        # Decode with 4x upsample (ph -> 4ph)
        x = self.decoder(x)  # [B*S, 64, 4ph, 4pw]

        # Upsample to original image resolution
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        # Final 1x1 conv to logits
        x = self.output_conv(x)  # [B*S, 1, H, W]

        # Remove channel dim and restore batch/sequence dims
        x = x.squeeze(1).reshape(B, S, H, W)

        return x
