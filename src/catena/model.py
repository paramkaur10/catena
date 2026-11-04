"""Catena's network: a band encoder for any set of bands, and a U-Net tuned by the sensor.

Band encoder:
  1. each band's descriptor (15 numbers; 14 before the pixel size joined) becomes a vector of
     64 (a small MLP);
  2. one Transformer layer lets the bands of an image attend to each other (padding masked);
  3. every band image goes through the same two 3x3 convolutions, scaled and shifted per
     channel by its band vector (FiLM);
  4. pooling over bands: a weighted mean (weights from the band vectors) and a maximum, so
     3 bands or 13, in any order, give the same 32 channels;
  5. the sensor vector z is the mean of the band vectors.
U-Net: 4 levels of 32, 64, 128 and 256 channels; every block is conv 3x3 -> GroupNorm ->
FiLM(z) -> GELU; a 1x1 head gives 4 outputs per pixel: clear, thin, thick, shadow. Height
and width must be multiples of 8.

FiLM starts at zero (gamma 1, beta 0), so training starts from a plain U-Net. For one fixed
sensor every FiLM is a constant that fold_block() bakes into the GroupNorm before flight.
2,185,141 parameters with the default sizes; the budget is 3.75 M.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from catena.descriptor import N_FIELDS

EMB = 64


class FiLM(nn.Module):
    """y = (1 + dg) * x + db per channel, with dg and db computed from a vector z.
    Starts at zero, so a new FiLM layer changes nothing until training gives it a use."""

    def __init__(self, channels: int, cond: int = EMB):
        super().__init__()
        self.to_gb = nn.Linear(cond, 2 * channels)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, x, z):
        dg, db = self.to_gb(z).chunk(2, dim=-1)  # (B, C) each
        return x * (1 + dg[..., None, None]) + db[..., None, None]


class ConvFiLM(nn.Module):
    """3x3 conv -> GroupNorm -> FiLM(z) -> GELU: one block of the U-Net."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, cout)
        self.film = FiLM(cout)

    def forward(self, x, z):
        return F.gelu(self.film(self.norm(self.conv(x)), z))


class BandEncoder(nn.Module):
    """Any number of bands, each with its descriptor -> one feature map and a sensor vector."""

    def __init__(self, width: int = 16, fields: int = N_FIELDS):
        super().__init__()
        self.fields = fields  # a network trained with fewer fields reads only the first ones
        self.embed = nn.Sequential(nn.Linear(fields, EMB), nn.GELU(), nn.Linear(EMB, EMB))
        self.mix = nn.TransformerEncoderLayer(
            EMB, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True
        )  # attention across the bands of one image
        self.stem1 = nn.Conv2d(1, width, 3, padding=1)
        self.stem2 = nn.Conv2d(width, width, 3, padding=1)
        self.film1, self.film2 = FiLM(width), FiLM(width)
        self.score = nn.Linear(EMB, 1)

    def forward(self, x, d, real):
        """x (B, N, H, W) bands; d (B, N, fields or more) descriptors; real (B, N) bool, False
        = padding."""
        b, n, h, w = x.shape
        e = self.mix(self.embed(d[..., : self.fields]), src_key_padding_mask=~real)  # (B, N, EMB)
        e = e.masked_fill(~real[..., None], 0.0)  # padding slots carry nothing
        eb = e.reshape(b * n, EMB)
        f = x.reshape(b * n, 1, h, w)
        f = F.gelu(self.film1(self.stem1(f), eb))
        f = F.gelu(self.film2(self.stem2(f), eb)).reshape(b, n, -1, h, w)
        weight = self.score(e).squeeze(-1).masked_fill(~real, float("-inf")).softmax(dim=1)
        mean = (weight.to(f.dtype)[:, :, None, None, None] * f).sum(dim=1)  # (B, C, H, W)
        peak = f.masked_fill(~real[:, :, None, None, None], float("-inf")).amax(dim=1)
        sensor = (e * real[..., None]).sum(dim=1) / real.sum(dim=1, keepdim=True)  # (B, EMB)
        return torch.cat([mean, peak], dim=1), sensor


class CatenaNet(nn.Module):
    def __init__(
        self, widths=(32, 64, 128, 256), classes: int = 4, stem: int = 16, fields: int = N_FIELDS
    ):
        super().__init__()
        self.bands = BandEncoder(stem, fields)
        ins = [2 * stem, *widths[:-1]]
        self.down = nn.ModuleList(
            nn.ModuleList([ConvFiLM(i, o), ConvFiLM(o, o)])
            for i, o in zip(ins, widths, strict=True)
        )
        self.up = nn.ModuleList(
            nn.ModuleList([ConvFiLM(deep + o, o), ConvFiLM(o, o)])
            for deep, o in zip(widths[:0:-1], widths[-2::-1], strict=True)
        )
        self.head = nn.Conv2d(widths[0], classes, 1)

    def forward(self, x, d, real):
        f, z = self.bands(x, d, real)
        skips = []
        for i, (first, second) in enumerate(self.down):
            f = second(first(F.max_pool2d(f, 2) if i else f, z), z)
            skips.append(f)
        skips.pop()  # the deepest map is f itself
        for first, second in self.up:
            f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
            f = second(first(torch.cat([f, skips.pop()], dim=1), z), z)
        return self.head(f)  # (B, 4, H, W) logits; H and W must be multiples of 8


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


class NoFiLM(nn.Module):
    def forward(self, x, z):
        return x


@torch.no_grad()
def fold_block(block: ConvFiLM, z: torch.Tensor) -> None:
    """For one fixed sensor (z of shape (1, EMB)): bake FiLM into the GroupNorm's own
    scale and shift. g * (w * n + b) + db = (g * w) * n + (g * b + db)."""
    dg, db = block.film.to_gb(z).chunk(2, dim=-1)
    g = 1 + dg[0]
    block.norm.bias.mul_(g).add_(db[0])
    block.norm.weight.mul_(g)
    block.film = NoFiLM()


if __name__ == "__main__":  # shape and size check, about a second on a CPU
    net = CatenaNet().eval()
    x, d = torch.rand(2, 5, 64, 64), torch.randn(2, 5, N_FIELDS)
    real = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])  # second image: 3 bands
    with torch.no_grad():
        out = net(x, d, real)
        perm = torch.tensor([2, 0, 1, 3, 4])  # the same bands in another order
        same = net(x[:, perm], d[:, perm], real[:, perm])
    assert out.shape == (2, 4, 64, 64), out.shape
    print("order of bands changes nothing:", torch.allclose(out, same, atol=1e-5))
    print(f"{count_parameters(net):,} parameters")
