"""
Ternary Quantization & Multiplication-Free 2-Bit LUT Primitives for `SimVQ-Split` (`Calcite-5`):
- `quantize_ternary_weights_ls` / `quantize_ternary_ste`
- `PackedLUT2BitLinear` / `pack_model_2bit_lut` / `compute_2bit_lut_footprint`
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_ternary_weights_ls(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BitNet-b1.58 32-channel groupwise 2-step Least-Squares ternary quantization
    (`{-1, 0, +1} * scale`), returning `(w_tern, trits_int8, scale_1d)`.
    """
    orig_shape = w.shape
    g_size = 32 if w.numel() % 32 == 0 else w.size(-1)
    with torch.no_grad():
        w_g = w.detach().float().reshape(-1, g_size)
        s0 = w_g.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        t = torch.where(w_g.abs() >= 0.5 * s0, w_g.sign(), torch.zeros_like(w_g))
        for _ in range(2):
            denom = t.square().sum(dim=-1, keepdim=True)
            s_ls = torch.where(
                denom > 0.0,
                (w_g * t).sum(dim=-1, keepdim=True) / denom.clamp(min=1.0),
                s0,
            ).clamp(min=1e-6)
            t = torch.where(
                w_g.abs() >= 0.5 * s_ls, w_g.sign(), torch.zeros_like(w_g)
            )
        denom = t.square().sum(dim=-1, keepdim=True)
        scale = torch.where(
            denom > 0.0,
            (w_g * t).sum(dim=-1, keepdim=True) / denom.clamp(min=1.0),
            s0,
        ).clamp(min=1e-6)
        w_tern = (t * scale).reshape(orig_shape).to(w.dtype)
        return w_tern, t.to(torch.int8), scale.squeeze(-1)


def quantize_ternary_ste(w: torch.Tensor) -> torch.Tensor:
    """
    BitNet-b1.58 LS-optimal 32-channel groupwise ternary quantization (`{-1, 0, +1} * scale`)
    with Straight-Through Estimator (`w + (w_tern - w).detach()`).
    """
    w_tern, _, _ = quantize_ternary_weights_ls(w)
    return w + (w_tern - w).detach()


class PackedLUT2BitLinear(nn.Module):
    """
    Physical T-MAC 2-bit uint8 base-3 LUT Linear layer (4 ternary weights per byte = 2.0 bits/weight).
    Forward executes addition/subtraction-only 81-entry LUT synthesis + torch.gather + group scales
    with ZERO ternary weight multiplications.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_u8: torch.Tensor,
        raw_scales_fp16: torch.Tensor,
        bias_fp16: torch.Tensor | None,
        g_size: int,
        row_aligned: bool,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.g_size = int(g_size)
        self.row_aligned = bool(row_aligned)
        self.in_pad = ((self.in_features + 3) // 4) * 4
        self.k_bytes = self.in_pad // 4
        self.register_buffer("packed_u8", packed_u8.contiguous())
        self.register_buffer("raw_scales_fp16", raw_scales_fp16.contiguous())
        if self.row_aligned:
            g32 = self.in_features // 32
            self.register_buffer(
                "scales_fp16",
                self.raw_scales_fp16.view(self.out_features, g32).contiguous(),
            )
        else:
            self.register_buffer("scales_fp16", None)
        if bias_fp16 is not None:
            self.register_buffer("bias_fp16", bias_fp16.contiguous())
        else:
            self.register_buffer("bias_fp16", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "PackedLUT2BitLinear":
        w = lin.weight.detach()
        out_f, in_f = int(w.size(0)), int(w.size(1))
        g_size = 32 if w.numel() % 32 == 0 else in_f
        _, trits_i8, scale_1d = quantize_ternary_weights_ls(w)
        raw_scales_fp16 = scale_1d.to(torch.float16).reshape(-1)
        bias_fp16 = lin.bias.detach().to(torch.float16) if lin.bias is not None else None

        in_pad = ((in_f + 3) // 4) * 4
        k_bytes = in_pad // 4
        row_aligned = (in_f % 32 == 0) and (g_size == 32)

        t_full = trits_i8.reshape(out_f, in_f).to(torch.int16)
        if in_pad > in_f:
            t_full = F.pad(t_full, (0, in_pad - in_f), value=0)
        t4 = t_full.view(out_f, k_bytes, 4) + 1
        codes = (
            t4[..., 0] + 3 * t4[..., 1] + 9 * t4[..., 2] + 27 * t4[..., 3]
        ).to(torch.uint8)
        if row_aligned:
            packed_u8 = codes.view(out_f, in_f // 32, 8)
        else:
            packed_u8 = codes
        return cls(
            in_features=in_f,
            out_features=out_f,
            packed_u8=packed_u8,
            raw_scales_fp16=raw_scales_fp16,
            bias_fp16=bias_fp16,
            g_size=g_size,
            row_aligned=row_aligned,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        n_tot = x_flat.size(0)
        if n_tot == 0:
            return torch.empty(
                orig_shape + (self.out_features,), device=x.device, dtype=x.dtype
            )
        if self.in_pad > self.in_features:
            x_flat = F.pad(x_flat, (0, self.in_pad - self.in_features))

        outs = []
        n_blk = 256
        k_bytes = self.k_bytes
        if not self.row_aligned:
            row_o = torch.arange(self.out_features, device=x.device).unsqueeze(1)
            col_b = torch.arange(k_bytes, device=x.device).unsqueeze(0) * 4
            col_c = col_b.clamp(max=max(0, self.in_features - 1))
            b_idx = ((row_o * self.in_features + col_c) // self.g_size).clamp(
                max=self.raw_scales_fp16.numel() - 1
            )
            byte_scales = self.raw_scales_fp16[b_idx].t().float().unsqueeze(-1)
            idx_base = self.packed_u8.t().long().unsqueeze(-1)
        else:
            g32 = self.in_features // 32
            scales_grp = self.scales_fp16.t().float().unsqueeze(-1)
            idx_base = self.packed_u8.view(self.out_features, k_bytes).t().long().unsqueeze(-1)

        for start in range(0, n_tot, n_blk):
            xb = x_flat[start : start + n_blk].float()
            nb = xb.size(0)
            x4 = xb.view(nb, k_bytes, 4)

            lut = torch.zeros(nb, k_bytes, 1, device=x.device, dtype=torch.float32)
            for k in range(4):
                xk = x4[:, :, k : k + 1]
                lut = torch.cat([lut - xk, lut, lut + xk], dim=-1)

            lut_p = lut.permute(1, 2, 0)
            idx = idx_base.expand(-1, -1, nb)
            gathered = torch.gather(lut_p, dim=1, index=idx)

            if self.row_aligned:
                grp_sum = gathered.view(g32, 8, self.out_features, nb).sum(dim=1)
                y_b = (grp_sum * scales_grp).sum(dim=0).t()
            else:
                y_b = (gathered * byte_scales).sum(dim=0).t()
            outs.append(y_b)

        y = torch.cat(outs, dim=0)
        if self.bias_fp16 is not None:
            y = y + self.bias_fp16.float().unsqueeze(0)
        return y.to(x.dtype).view(orig_shape + (self.out_features,))


def pack_model_2bit_lut(model: nn.Module) -> nn.Module:
    """
    Deep-copies `model` in eval mode and replaces every ternary-quantized `nn.Linear`
    (keeping `lm_head` tied to `tok_emb`) with a physical `PackedLUT2BitLinear` T-MAC layer.
    """
    packed = copy.deepcopy(model)
    packed.train(False)
    lm_head_ref = getattr(packed, "lm_head", None)

    def _replace_linear(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                if child is lm_head_ref or name == "lm_head":
                    continue
                setattr(mod, name, PackedLUT2BitLinear.from_linear(child))
            else:
                _replace_linear(child)

    _replace_linear(packed)
    return packed


def compute_2bit_lut_footprint(lut_model: nn.Module) -> dict[str, float]:
    """
    Measures exact serialized footprint of a packed `2-bit uint8 T-MAC LUT` model:
    packed `uint8` weight matrices, `FP16` group scales/biases, and `FP16` embeddings/norms.
    """
    packed_u8_bytes = 0
    scales_fp16_bytes = 0
    orig_fp32_matrix_bytes = 0
    packed_weights_count = 0
    num_lut_layers = 0

    for m in lut_model.modules():
        if isinstance(m, PackedLUT2BitLinear):
            num_lut_layers += 1
            packed_u8_bytes += int(m.packed_u8.numel() * m.packed_u8.element_size())
            scales_fp16_bytes += int(m.raw_scales_fp16.numel() * 2)
            if m.bias_fp16 is not None:
                scales_fp16_bytes += int(m.bias_fp16.numel() * 2)
            n_w = int(m.out_features * m.in_features)
            packed_weights_count += n_w
            orig_fp32_matrix_bytes += n_w * 4

    seen_ids: list[int] = []
    other_params = 0
    for p in lut_model.parameters():
        pid = id(p)
        if pid not in seen_ids:
            seen_ids.append(pid)
            other_params += int(p.numel())
    fp16_other_bytes = other_params * 2
    total_bytes = packed_u8_bytes + scales_fp16_bytes + fp16_other_bytes

    return dict(
        num_lut_layers=int(num_lut_layers),
        packed_weights_count=int(packed_weights_count),
        packed_u8_bytes=int(packed_u8_bytes),
        packed_u8_kb=round(packed_u8_bytes / 1024.0, 2),
        scales_fp16_bytes=int(scales_fp16_bytes),
        scales_fp16_kb=round(scales_fp16_bytes / 1024.0, 2),
        fp16_other_bytes=int(fp16_other_bytes),
        fp16_other_kb=round(fp16_other_bytes / 1024.0, 2),
        total_lut_kb=round(total_bytes / 1024.0, 2),
        orig_fp32_matrix_kb=round(orig_fp32_matrix_bytes / 1024.0, 2),
        matrix_compression_vs_fp32=round(
            orig_fp32_matrix_bytes / max(1, packed_u8_bytes), 4
        ),
        effective_matrix_compression_with_scales=round(
            orig_fp32_matrix_bytes / max(1, packed_u8_bytes + scales_fp16_bytes), 4
        ),
    )
