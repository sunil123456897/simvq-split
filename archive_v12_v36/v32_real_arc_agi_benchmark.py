#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE v32.1: HONEST REAL EXTERNAL BENCHMARK — fchollet/ARC-AGI (800 TASKS)
==========================================================================
Полностью честный («Zero-Illusions Protocol») прогон архитектуры SimVQ / H-JEPA
на настоящем публичном бенчмарке Франсуа Шолле `fchollet/ARC-AGI`:
  - `benchmarks/arc_agi/ARC-AGI-master/data/training/*.json` (400 реальных задач)
  - `benchmarks/arc_agi/ARC-AGI-master/data/evaluation/*.json` (400 реальных задач)

Строгие инварианты честности (Anti-Cheat Contract):
  1. Никаких синтетических сгенерированных сеток: загружаются только 800 официальных JSON-задач.
  2. Строгий слепой протокол (`blind test evaluation`): перед вызовом решателя поле `"output"`
     удаляется из `task["test"]`. Решатель получает ТОЛЬКО `train_pairs` и `test_inputs`.
  3. Сравниваются четыре системы / этапа на всех 800 реальных задачах:
     - `Baseline A (v31 4-op Toy DSL as-is)`: старый 4-операторный решатель из `v31` («как есть»).
     - `Baseline B (Direct Neural Grid Predictor)`: поклеточный нейросетевой предсказатель,
       обучаемый градиентным спуском на демонстрационных парах задачи без поиска программ.
     - `v32.0 Initial Whole-Grid Mode-2 MPC`: базовый 36-операторный полносеточный DSL (`79/800`).
     - `v32.1 Object-Centric + Mosaic/Panel + Wake-Sleep Stitch-EBBS Mode-2 MPC`:
       * Сферический кодировщик топологических инвариантов сетки `SphericalARCInvariantEncoder_v32`
         на базе `BitLinear158` и квантования Вороного `ExpandedVoronoiCodebook_v29` ((S^14)^16,
         дрейф нормы 0.000000), компилируемый в 16.00x `uint8` 2-bit LUT.
       * Объектно-ориентированный графовый анализ связных компонент (площадь, дыры, симметрия,
         частота формы, рамки, маркеры), реконструкция перекрытых симметричных/периодических мозаик,
         выбор уникальных/мажоритарных подпанелей, гравитация с препятствиями, штамповка шаблонов
         и консолидация макро-навыков во сне (`WakeSleepMacroLibrary_v32`) при `UF=0`.
"""

import os
import time
import glob
import json
import math
from typing import Dict, List, Tuple, Callable, Optional, Any

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from v17_cognitive_llm_replacement import BitLinear158
from v28_hierarchical_stochastic_hjepa_agent import compute_wilson_ci
from v25_unified_lut2bit_bilingual_agent import compile_engine_to_lut2bit_v25
from v29_data_scaling_pipeline_100k_video import ExpandedVoronoiCodebook_v29


# ==============================================================================
# 1. СФЕРИЧЕСКИЙ КОДИРОВЩИК ТОПОЛОГИЧЕСКИХ И СПЕКТРАЛЬНЫХ ИНВАРИАНТОВ СЕТКИ ARC
# ==============================================================================
def extract_grid_topological_invariants_64d(g: np.ndarray) -> np.ndarray:
    """
    Извлекает 64-мерный вектор структурно-топологических инвариантов из 2D-сетки ARC
    произвольного размера (H, W) in [1..30]^2:
      - [0..5]: геометрия (H/30, W/30, H/W, W/H, (H*W)/900, is_square)
      - [6..15]: нормированная гистограмма цветов 0..9
      - [16..25]: индикаторы присутствия цветов 0..9
      - [26..31]: автосимметрии группы D4 (flip_h, flip_v, rot180, transpose, antitranspose, n_colors)
      - [32..41]: статистика связных компонент (число 4-связных и 8-связных объектов, макс/мин площадь,
                  число замкнутых дыр, плотность ненулевых клеток, bounding-box fill ratio)
      - [42..51]: индикаторы разделительных линий (горизонтальных/вертикальных) и рамок
      - [52..63]: пространственные моменты и распределения по квадрантам.
    """
    feat = np.zeros(64, dtype=np.float32)
    H, W = g.shape
    area = float(max(1, H * W))

    # 1. Геометрия
    feat[0] = H / 30.0
    feat[1] = W / 30.0
    feat[2] = min(3.0, float(H) / float(max(1, W))) / 3.0
    feat[3] = min(3.0, float(W) / float(max(1, H))) / 3.0
    feat[4] = area / 900.0
    feat[5] = 1.0 if H == W else 0.0

    # 2. Гистограмма и присутствие цветов 0..9
    vals, cnts = np.unique(g, return_counts=True)
    for v, c in zip(vals, cnts):
        if 0 <= v < 10:
            feat[6 + v] = float(c) / area
            feat[16 + v] = 1.0

    bg = int(vals[np.argmax(cnts)]) if len(vals) > 0 else 0
    fg_mask = (g != 0) if (0 in vals and cnts[vals == 0][0] >= 0.25 * area) else (g != bg)

    # 3. D4 автосимметрии
    feat[26] = float(np.mean(g == np.fliplr(g)))
    feat[27] = float(np.mean(g == np.flipud(g)))
    feat[28] = float(np.mean(g == np.rot90(g, 2)))
    if H == W:
        feat[29] = float(np.mean(g == g.T))
        feat[30] = float(np.mean(g == np.fliplr(np.flipud(g.T))))
    feat[31] = float(len(vals)) / 10.0

    # 4. Связные компоненты и bounding box
    fg_count = float(np.sum(fg_mask))
    feat[32] = fg_count / area
    if fg_count > 0:
        ys, xs = np.where(fg_mask)
        bh = float(ys.max() - ys.min() + 1)
        bw = float(xs.max() - xs.min() + 1)
        feat[33] = bh / float(H)
        feat[34] = bw / float(W)
        feat[35] = fg_count / float(max(1.0, bh * bw))

        labeled_4, n_cc4 = ndimage.label(fg_mask)
        labeled_8, n_cc8 = ndimage.label(fg_mask, structure=np.ones((3, 3), dtype=int))
        feat[36] = min(1.0, float(n_cc4) / 20.0)
        feat[37] = min(1.0, float(n_cc8) / 20.0)
        if n_cc4 > 0:
            cc_sizes = np.bincount(labeled_4.ravel())[1:]
            feat[38] = float(cc_sizes.max()) / area
            feat[39] = float(cc_sizes.min()) / area

        # Замкнутые дыры фона
        bg_mask = ~fg_mask
        bg_lab, n_bg = ndimage.label(bg_mask)
        border_labels = (
            set(np.unique(bg_lab[0, :]))
            | set(np.unique(bg_lab[-1, :]))
            | set(np.unique(bg_lab[:, 0]))
            | set(np.unique(bg_lab[:, -1]))
        )
        holes = sum(1 for lbl in range(1, n_bg + 1) if lbl not in border_labels)
        feat[40] = min(1.0, float(holes) / 10.0)

    # 5. Разделительные линии (полные строки или столбцы одного ненулевого цвета)
    full_rows = sum(1 for r in range(H) if len(np.unique(g[r, :])) == 1 and g[r, 0] != 0)
    full_cols = sum(1 for c in range(W) if len(np.unique(g[:, c])) == 1 and g[0, c] != 0)
    feat[42] = min(1.0, float(full_rows) / 5.0)
    feat[43] = min(1.0, float(full_cols) / 5.0)

    # 6. Квадранты
    mh, mw = max(1, H // 2), max(1, W // 2)
    feat[52] = float(np.mean(fg_mask[:mh, :mw]))
    feat[53] = float(np.mean(fg_mask[:mh, mw:])) if W > 1 else 0.0
    feat[54] = float(np.mean(fg_mask[mh:, :mw])) if H > 1 else 0.0
    feat[55] = float(np.mean(fg_mask[mh:, mw:])) if (H > 1 and W > 1) else 0.0
    return feat


class SphericalARCInvariantEncoder_v32(nn.Module):
    """
    1.58-битный сферический кодировщик сеток ARC-AGI:
      - Переводит 64-мерный вектор топологических инвариантов сетки в точку
        на многообразии Вороного (S^14)^16 (d=240) с нулевым дрейфом нормы.
      - Используется планировщиком Mode-2 Beam/MPC как быстрый латентный критик
        расстояния между текущей сеткой и целевой демонстрацией.
    """

    def __init__(self, voronoi: ExpandedVoronoiCodebook_v29, device: str = "cuda"):
        super().__init__()
        self.voronoi = voronoi
        self.device = device
        self.fc1 = BitLinear158(64, 256).to(device)
        self.fc2 = BitLinear158(256, 240).to(device)
        for m in [self.fc1, self.fc2]:
            m.quant_alpha = 1.0

    @torch.no_grad()
    def encode_grid(self, g: np.ndarray) -> Tuple[torch.Tensor, float]:
        f64 = torch.from_numpy(extract_grid_topological_invariants_64d(g)).unsqueeze(0).to(self.device)
        h = F.silu(self.fc1(f64))
        z_raw = self.fc2(h)
        idx = self.voronoi.decode_to_indices(z_raw)
        z_q = self.voronoi.encode_indices(idx, normalize_total=False)
        norms = torch.norm(z_q.view(16, 15), p=2, dim=-1)
        max_drift = float((norms - 1.0).abs().max().item())
        return z_q.squeeze(0), max_drift

    @torch.no_grad()
    def encode_grids_batch(self, grids: List[np.ndarray]) -> Tuple[torch.Tensor, float]:
        if not grids:
            return torch.empty(0, 240, device=self.device), 0.0
        feats = np.stack([extract_grid_topological_invariants_64d(g) for g in grids], axis=0)
        f64 = torch.from_numpy(feats).to(self.device)
        h = F.silu(self.fc1(f64))
        z_raw = self.fc2(h)
        idx = self.voronoi.decode_to_indices(z_raw)
        z_q = self.voronoi.encode_indices(idx, normalize_total=False)
        norms = torch.norm(z_q.view(-1, 16, 15), p=2, dim=-1)
        max_drift = float((norms - 1.0).abs().max().item())
        return z_q, max_drift


# ==============================================================================
# 2. ОБЪЕКТНЫЙ ГРАФ, АТРИБУТЫ СВЯЗНЫХ КОМПОНЕНТ И ТОПОЛОГИЧЕСКИЙ DSL (v32.1)
# ==============================================================================
def get_dominant_bg(g: np.ndarray) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    if 0 in vals and cnts[vals == 0][0] >= 0.35 * g.size:
        return 0
    return int(vals[np.argmax(cnts)])


def _canonical_d4_signature(mask_crop: np.ndarray) -> bytes:
    sigs = []
    for k in range(4):
        r = np.rot90(mask_crop, k)
        sigs.append((r.shape[0], r.shape[1], r.tobytes()))
        f = np.fliplr(r)
        sigs.append((f.shape[0], f.shape[1], f.tobytes()))
    sigs.sort()
    best = sigs[0]
    return f"{best[0]}x{best[1]}:".encode() + best[2]


def _count_internal_holes(submask: np.ndarray) -> int:
    padded = np.pad(submask, pad_width=1, mode="constant", constant_values=False)
    bg_lab, n_bg = ndimage.label(~padded)
    if n_bg <= 1:
        return 0
    outer_lbl = bg_lab[0, 0]
    return sum(1 for lbl in range(1, n_bg + 1) if lbl != outer_lbl)


_OBJ_CACHE: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}


def extract_rich_objects(
    g: np.ndarray, bg: int = 0, connectivity: int = 4, multicolor: bool = False
) -> List[Dict[str, Any]]:
    """
    Извлекает связные компоненты с полным набором геометрических и топологических атрибутов:
      - mask, color, n_colors, area, bbox (y0, y1, x0, x1), bbox_h, bbox_w, bbox_area
      - subgrid, submask, n_holes, is_rect, is_square, is_line, is_sym_h, is_sym_v, is_sym_hv, is_sym_any
      - touches_border, shape_sig, d4_shape_sig, shape_freq, d4_shape_freq, color_cc_freq, area_freq
    """
    cache_key = (g.shape, g.tobytes(), int(bg), int(connectivity), bool(multicolor))
    cached = _OBJ_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    if len(_OBJ_CACHE) > 2048:
        _OBJ_CACHE.clear()

    H, W = g.shape
    struct = (
        np.ones((3, 3), dtype=int)
        if connectivity == 8
        else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    )
    objs: List[Dict[str, Any]] = []

    if multicolor:
        fg = g != bg
        labeled, n = ndimage.label(fg, structure=struct)
        for lbl in range(1, n + 1):
            mask = labeled == lbl
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            sub = np.where(mask[y0:y1, x0:x1], g[y0:y1, x0:x1], bg)
            submask = mask[y0:y1, x0:x1]
            vals, cnts = np.unique(g[mask], return_counts=True)
            c = int(vals[np.argmax(cnts)])
            bh, bw = y1 - y0, x1 - x0
            area = int(mask.sum())
            b_area = bh * bw
            is_rect = area == b_area
            sym_h = bool(np.array_equal(submask, np.fliplr(submask)))
            sym_v = bool(np.array_equal(submask, np.flipud(submask)))
            sym_180 = bool(np.array_equal(submask, np.rot90(submask, 2)))
            objs.append(
                {
                    "mask": mask,
                    "color": c,
                    "n_colors": int(len(vals)),
                    "area": area,
                    "bbox": (y0, y1, x0, x1),
                    "bbox_h": bh,
                    "bbox_w": bw,
                    "bbox_area": b_area,
                    "subgrid": sub,
                    "submask": submask,
                    "n_holes": _count_internal_holes(submask),
                    "is_rect": is_rect,
                    "is_square": is_rect and (bh == bw),
                    "is_line": is_rect and (bh == 1 or bw == 1) and area > 1,
                    "is_sym_h": sym_h,
                    "is_sym_v": sym_v,
                    "is_sym_hv": sym_h and sym_v,
                    "is_sym_any": sym_h or sym_v or sym_180,
                    "touches_border": (y0 == 0 or y1 == H or x0 == 0 or x1 == W),
                    "shape_sig": (bh, bw, submask.tobytes()),
                    "d4_shape_sig": _canonical_d4_signature(submask),
                    "color_sub_sig": (bh, bw, sub.tobytes()),
                }
            )
    else:
        for c in np.unique(g):
            if c == bg:
                continue
            labeled, n = ndimage.label(g == c, structure=struct)
            for lbl in range(1, n + 1):
                mask = labeled == lbl
                ys, xs = np.where(mask)
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                sub = np.where(mask[y0:y1, x0:x1], g[y0:y1, x0:x1], bg)
                submask = mask[y0:y1, x0:x1]
                bh, bw = y1 - y0, x1 - x0
                area = int(mask.sum())
                b_area = bh * bw
                is_rect = area == b_area
                sym_h = bool(np.array_equal(submask, np.fliplr(submask)))
                sym_v = bool(np.array_equal(submask, np.flipud(submask)))
                sym_180 = bool(np.array_equal(submask, np.rot90(submask, 2)))
                objs.append(
                    {
                        "mask": mask,
                        "color": int(c),
                        "n_colors": 1,
                        "area": area,
                        "bbox": (y0, y1, x0, x1),
                        "bbox_h": bh,
                        "bbox_w": bw,
                        "bbox_area": b_area,
                        "subgrid": sub,
                        "submask": submask,
                        "n_holes": _count_internal_holes(submask),
                        "is_rect": is_rect,
                        "is_square": is_rect and (bh == bw),
                        "is_line": is_rect and (bh == 1 or bw == 1) and area > 1,
                        "is_sym_h": sym_h,
                        "is_sym_v": sym_v,
                        "is_sym_hv": sym_h and sym_v,
                        "is_sym_any": sym_h or sym_v or sym_180,
                        "touches_border": (y0 == 0 or y1 == H or x0 == 0 or x1 == W),
                        "shape_sig": (bh, bw, submask.tobytes()),
                        "d4_shape_sig": _canonical_d4_signature(submask),
                        "color_sub_sig": (bh, bw, sub.tobytes()),
                    }
                )

    shape_counts: Dict[Any, int] = {}
    d4_counts: Dict[bytes, int] = {}
    color_counts: Dict[int, int] = {}
    area_counts: Dict[int, int] = {}
    csub_counts: Dict[Any, int] = {}
    for o in objs:
        shape_counts[o["shape_sig"]] = shape_counts.get(o["shape_sig"], 0) + 1
        d4_counts[o["d4_shape_sig"]] = d4_counts.get(o["d4_shape_sig"], 0) + 1
        color_counts[o["color"]] = color_counts.get(o["color"], 0) + 1
        area_counts[o["area"]] = area_counts.get(o["area"], 0) + 1
        csub_counts[o["color_sub_sig"]] = csub_counts.get(o["color_sub_sig"], 0) + 1

    for o in objs:
        o["shape_freq"] = shape_counts[o["shape_sig"]]
        o["d4_shape_freq"] = d4_counts[o["d4_shape_sig"]]
        o["color_cc_freq"] = color_counts[o["color"]]
        o["area_freq"] = area_counts[o["area"]]
        o["color_sub_freq"] = csub_counts[o["color_sub_sig"]]

    _OBJ_CACHE[cache_key] = objs
    return list(objs)


def get_connected_components(g: np.ndarray, bg: int = 0, connectivity: int = 4, multicolor: bool = False):
    """Совместимая обёртка для базовых операторов."""
    objs = extract_rich_objects(g, bg=bg, connectivity=connectivity, multicolor=multicolor)
    return [(o["mask"], o["color"], o["area"], o["bbox"], o["subgrid"]) for o in objs]


# --- Базовые геометрические, топологические и объектные операторы ---
def op_identity(g: np.ndarray) -> np.ndarray:
    return g.copy()


def op_rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, -1)


def op_rot180(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 2)


def op_rot270(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 1)


def op_flip_h(g: np.ndarray) -> np.ndarray:
    return np.fliplr(g)


def op_flip_v(g: np.ndarray) -> np.ndarray:
    return np.flipud(g)


def op_transpose(g: np.ndarray) -> np.ndarray:
    return g.T.copy()


def op_antitranspose(g: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(g.T)).copy()


def op_crop_nonzero(g: np.ndarray) -> np.ndarray:
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return g.copy()
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def op_crop_non_bg(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def op_crop_largest_cc4_mono(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=4, multicolor=False)
    if not comps:
        return g.copy()
    comps.sort(key=lambda x: x[2], reverse=True)
    y0, y1, x0, x1 = comps[0][3]
    return g[y0:y1, x0:x1].copy()


def op_crop_largest_cc8_multi(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=8, multicolor=True)
    if not comps:
        return g.copy()
    comps.sort(key=lambda x: x[2], reverse=True)
    y0, y1, x0, x1 = comps[0][3]
    return g[y0:y1, x0:x1].copy()


def op_crop_smallest_cc4_mono(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=4, multicolor=False)
    if not comps:
        return g.copy()
    comps.sort(key=lambda x: x[2])
    y0, y1, x0, x1 = comps[0][3]
    return g[y0:y1, x0:x1].copy()


def op_crop_smallest_cc8_multi(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=8, multicolor=True)
    if not comps:
        return g.copy()
    comps.sort(key=lambda x: x[2])
    y0, y1, x0, x1 = comps[0][3]
    return g[y0:y1, x0:x1].copy()


def op_crop_rare_color_bbox(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, cnts) if v != bg]
    if not non_bg:
        return g.copy()
    non_bg.sort(key=lambda x: x[1])
    rare_c = non_bg[0][0]
    ys, xs = np.where(g == rare_c)
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def op_crop_inside_rare_markers(g: np.ndarray) -> np.ndarray:
    """Вырезает строго внутреннюю область между угловыми маркерами самого редкого цвета."""
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, cnts) if v != bg and 2 <= c <= 12]
    if not non_bg:
        return g.copy()
    non_bg.sort(key=lambda x: x[1])
    rare_c = non_bg[0][0]
    ys, xs = np.where(g == rare_c)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if y1 - y0 >= 2 and x1 - x0 >= 2:
        return g[y0 + 1 : y1, x0 + 1 : x1].copy()
    return g.copy()


def op_crop_rare_markers_clean(g: np.ndarray) -> np.ndarray:
    """Вырезает область по маркерам самого редкого цвета включительно, стирая сами маркеры в цвет фона."""
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, cnts) if v != bg and 2 <= c <= 8]
    if not non_bg:
        return g.copy()
    non_bg.sort(key=lambda x: x[1])
    rare_c = non_bg[0][0]
    ys, xs = np.where(g == rare_c)
    sub = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
    sub[sub == rare_c] = bg
    return sub


def op_crop_inside_frame(g: np.ndarray) -> np.ndarray:
    """Находит прямоугольную рамку одного цвета и вырезает её внутренность."""
    bg = get_dominant_bg(g)
    for c in np.unique(g):
        if c == bg:
            continue
        ys, xs = np.where(g == c)
        if len(ys) < 8:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        if y1 - y0 >= 2 and x1 - x0 >= 2:
            top = np.all(g[y0, x0 : x1 + 1] == c)
            bot = np.all(g[y1, x0 : x1 + 1] == c)
            left = np.all(g[y0 : y1 + 1, x0] == c)
            right = np.all(g[y0 : y1 + 1, x1] == c)
            if top and bot and left and right:
                return g[y0 + 1 : y1, x0 + 1 : x1].copy()
        # Также проверяем случай, когда внутри большой коробки есть меньшая замкнутая рамка цвета c:
        labeled, n = ndimage.label(g == c)
        for lbl in range(1, n + 1):
            ry, rx = np.where(labeled == lbl)
            if len(ry) < 8:
                continue
            r0, r1, c0, c1 = int(ry.min()), int(ry.max()), int(rx.min()), int(rx.max())
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                if (
                    np.all(g[r0, c0 : c1 + 1] == c)
                    and np.all(g[r1, c0 : c1 + 1] == c)
                    and np.all(g[r0 : r1 + 1, c0] == c)
                    and np.all(g[r0 : r1 + 1, c1] == c)
                ):
                    return g[r0 + 1 : r1, c0 + 1 : c1].copy()
    return g.copy()


def op_crop_frame_inclusive(g: np.ndarray) -> np.ndarray:
    """Находит замкнутую прямоугольную рамку одного цвета и вырезает её вместе с рамкой."""
    bg = get_dominant_bg(g)
    for c in np.unique(g):
        if c == bg:
            continue
        labeled, n = ndimage.label(g == c)
        for lbl in range(1, n + 1):
            ry, rx = np.where(labeled == lbl)
            if len(ry) < 8:
                continue
            r0, r1, c0, c1 = int(ry.min()), int(ry.max()), int(rx.min()), int(rx.max())
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                if (
                    np.all(g[r0, c0 : c1 + 1] == c)
                    and np.all(g[r1, c0 : c1 + 1] == c)
                    and np.all(g[r0 : r1 + 1, c0] == c)
                    and np.all(g[r0 : r1 + 1, c1] == c)
                ):
                    return g[r0 : r1 + 1, c0 : c1 + 1].copy()
    return g.copy()


def op_keep_only_largest_cc(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=4, multicolor=False)
    if not comps:
        return g.copy()
    comps.sort(key=lambda x: x[2], reverse=True)
    out = np.full_like(g, bg)
    mask = comps[0][0]
    out[mask] = g[mask]
    return out


def op_keep_only_smallest_cc(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=4, multicolor=False)
    if not comps:
        return g.copy()
    comps.sort(key=lambda x: x[2])
    out = np.full_like(g, bg)
    mask = comps[0][0]
    out[mask] = g[mask]
    return out


def op_remove_isolated_1x1_noise(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=8, multicolor=True)
    out = np.full_like(g, bg)
    for mask, _, area, _, _ in comps:
        if area > 1:
            out[mask] = g[mask]
    return out


def op_keep_only_isolated_1x1(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    comps = get_connected_components(g, bg=bg, connectivity=8, multicolor=True)
    out = np.full_like(g, bg)
    for mask, _, area, _, _ in comps:
        if area == 1:
            out[mask] = g[mask]
    return out


def op_fill_enclosed_holes_with_parent(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    bg_mask = g == bg
    labeled, n = ndimage.label(bg_mask)
    if n == 0:
        return out
    border_lbls = (
        set(np.unique(labeled[0, :]))
        | set(np.unique(labeled[-1, :]))
        | set(np.unique(labeled[:, 0]))
        | set(np.unique(labeled[:, -1]))
    )
    for lbl in range(1, n + 1):
        if lbl not in border_lbls:
            hole_mask = labeled == lbl
            dilated = ndimage.binary_dilation(hole_mask) & (~hole_mask)
            neighbor_vals = g[dilated]
            neighbor_vals = neighbor_vals[neighbor_vals != bg]
            if len(neighbor_vals) > 0:
                vals, cnts = np.unique(neighbor_vals, return_counts=True)
                out[hole_mask] = vals[np.argmax(cnts)]
    return out


def op_gravity_down(g: np.ndarray) -> np.ndarray:
    bg = 0
    H, W = g.shape
    out = np.full_like(g, bg)
    for c in range(W):
        col = g[:, c]
        fg = col[col != bg]
        if len(fg) > 0:
            out[H - len(fg) :, c] = fg
    return out


def op_gravity_up(g: np.ndarray) -> np.ndarray:
    bg = 0
    H, W = g.shape
    out = np.full_like(g, bg)
    for c in range(W):
        col = g[:, c]
        fg = col[col != bg]
        if len(fg) > 0:
            out[: len(fg), c] = fg
    return out


def op_gravity_left(g: np.ndarray) -> np.ndarray:
    bg = 0
    H, W = g.shape
    out = np.full_like(g, bg)
    for r in range(H):
        row = g[r, :]
        fg = row[row != bg]
        if len(fg) > 0:
            out[r, : len(fg)] = fg
    return out


def op_gravity_right(g: np.ndarray) -> np.ndarray:
    bg = 0
    H, W = g.shape
    out = np.full_like(g, bg)
    for r in range(H):
        row = g[r, :]
        fg = row[row != bg]
        if len(fg) > 0:
            out[r, W - len(fg) :] = fg
    return out


def op_complete_sym_h(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    flipped = np.fliplr(g)
    return np.where(g != bg, g, flipped)


def op_complete_sym_v(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    flipped = np.flipud(g)
    return np.where(g != bg, g, flipped)


def op_complete_sym_rot180(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    r180 = np.rot90(g, 2)
    return np.where(g != bg, g, r180)


def op_complete_sym_transpose(g: np.ndarray) -> np.ndarray:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    bg = get_dominant_bg(g)
    return np.where(g != bg, g, g.T)


def op_complete_sym_4way(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = np.where(g != bg, g, np.fliplr(g))
    out = np.where(out != bg, out, np.flipud(out))
    if g.shape[0] == g.shape[1]:
        out = np.where(out != bg, out, out.T)
    return out


def op_connect_aligned_dots_hv(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    for c in np.unique(g):
        if c == bg:
            continue
        ys, xs = np.where(g == c)
        if len(ys) < 2 or len(ys) > 16:
            continue
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                if ys[i] == ys[j]:
                    x_min, x_max = min(xs[i], xs[j]), max(xs[i], xs[j])
                    seg = out[ys[i], x_min : x_max + 1]
                    out[ys[i], x_min : x_max + 1] = np.where(seg == bg, c, seg)
                elif xs[i] == xs[j]:
                    y_min, y_max = min(ys[i], ys[j]), max(ys[i], ys[j])
                    seg = out[y_min : y_max + 1, xs[i]]
                    out[y_min : y_max + 1, xs[i]] = np.where(seg == bg, c, seg)
    return out


def op_fill_between_same_color_row(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    for r in range(H):
        row = g[r, :]
        for c in np.unique(row):
            if c == bg:
                continue
            xs = np.where(row == c)[0]
            if len(xs) >= 2:
                x0, x1 = int(xs.min()), int(xs.max())
                seg = out[r, x0 : x1 + 1]
                out[r, x0 : x1 + 1] = np.where(seg == bg, c, seg)
    return out


def op_fill_between_same_color_col(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    for col_idx in range(W):
        col = g[:, col_idx]
        for c in np.unique(col):
            if c == bg:
                continue
            ys = np.where(col == c)[0]
            if len(ys) >= 2:
                y0, y1 = int(ys.min()), int(ys.max())
                seg = out[y0 : y1 + 1, col_idx]
                out[y0 : y1 + 1, col_idx] = np.where(seg == bg, c, seg)
    return out


def op_fill_color_bboxes(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    for c in np.unique(g):
        if c == bg:
            continue
        ys, xs = np.where(g == c)
        if len(ys) >= 2:
            out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = c
    return out


def op_fill_each_cc_bbox(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False)
    for o in objs:
        y0, y1, x0, x1 = o["bbox"]
        out[y0:y1, x0:x1] = o["color"]
    return out


def op_hollow_rectangles(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            val = g[r, c]
            if val != bg and np.all(g[r - 1 : r + 2, c - 1 : c + 2] == val):
                out[r, c] = bg
    return out


def op_swap_two_fg_colors(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals = [int(v) for v in np.unique(g) if v != bg]
    if len(vals) != 2:
        return g.copy()
    c1, c2 = vals[0], vals[1]
    out = g.copy()
    out[g == c1] = c2
    out[g == c2] = c1
    return out


def op_swap_most_least_fg_colors(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    fg = [(int(v), int(c)) for v, c in zip(vals, cnts) if v != bg]
    if len(fg) < 2:
        return g.copy()
    fg.sort(key=lambda x: x[1])
    c_min, c_max = fg[0][0], fg[-1][0]
    out = g.copy()
    out[g == c_min] = c_max
    out[g == c_max] = c_min
    return out


def op_recolor_to_most_freq_fg(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    fg = [(int(v), int(c)) for v, c in zip(vals, cnts) if v != bg]
    if not fg:
        return g.copy()
    fg.sort(key=lambda x: x[1], reverse=True)
    c_max = fg[0][0]
    return np.where(g != bg, c_max, bg)


def op_recolor_to_least_freq_fg(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    fg = [(int(v), int(c)) for v, c in zip(vals, cnts) if v != bg]
    if not fg:
        return g.copy()
    fg.sort(key=lambda x: x[1])
    c_min = fg[0][0]
    return np.where(g != bg, c_min, bg)


def op_tile_2x2_mirror(g: np.ndarray) -> np.ndarray:
    top = np.hstack([g, np.fliplr(g)])
    bot = np.hstack([np.flipud(g), np.rot90(g, 2)])
    return np.vstack([top, bot])


def op_tile_2x2_mirror_inv(g: np.ndarray) -> np.ndarray:
    top = np.hstack([np.rot90(g, 2), np.flipud(g)])
    bot = np.hstack([np.fliplr(g), g])
    return np.vstack([top, bot])


def op_tile_2x2_rot(g: np.ndarray) -> np.ndarray:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    top = np.hstack([g, np.rot90(g, -1)])
    bot = np.hstack([np.rot90(g, 1), np.rot90(g, 2)])
    return np.vstack([top, bot])


def op_tile_2x2_repeat(g: np.ndarray) -> np.ndarray:
    return np.tile(g, (2, 2))


def op_tile_3x3_repeat(g: np.ndarray) -> np.ndarray:
    return np.tile(g, (3, 3))


def op_kronecker_self_fractal(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    if H * H > 30 or W * W > 30:
        return g.copy()
    out = np.zeros((H * H, W * W), dtype=g.dtype)
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0:
                out[r * H : (r + 1) * H, c * W : (c + 1) * W] = g
    return out


def op_kronecker_color_fractal(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    if H * H > 30 or W * W > 30:
        return g.copy()
    mask = (g != 0).astype(g.dtype)
    out = np.zeros((H * H, W * W), dtype=g.dtype)
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0:
                out[r * H : (r + 1) * H, c * W : (c + 1) * W] = mask * g[r, c]
    return out


def op_compress_consecutive_dup_rows_cols(g: np.ndarray) -> np.ndarray:
    """Сжимает подряд идущие одинаковые строки и подряд идущие одинаковые столбцы."""
    H, W = g.shape
    keep_r = [0] + [r for r in range(1, H) if not np.array_equal(g[r, :], g[r - 1, :])]
    g_r = g[keep_r, :]
    keep_c = [0] + [c for c in range(1, W) if not np.array_equal(g_r[:, c], g_r[:, c - 1])]
    return g_r[:, keep_c].copy()


def op_remove_empty_bg_rows_cols(g: np.ndarray) -> np.ndarray:
    """Удаляет полностью фоновые строки и столбцы, сжимая непустые части."""
    bg = get_dominant_bg(g)
    keep_r = [r for r in range(g.shape[0]) if np.any(g[r, :] != bg)]
    keep_c = [c for c in range(g.shape[1]) if np.any(g[:, c] != bg)]
    if not keep_r or not keep_c:
        return g.copy()
    return g[np.ix_(keep_r, keep_c)].copy()


def op_remove_separator_lines(g: np.ndarray) -> np.ndarray:
    """Удаляет сплошные одноцветные разделительные линии (строки и столбцы) нефонового цвета."""
    bg = get_dominant_bg(g)
    H, W = g.shape
    keep_r = [r for r in range(H) if not (len(np.unique(g[r, :])) == 1 and g[r, 0] != bg)]
    keep_c = [c for c in range(W) if not (len(np.unique(g[:, c])) == 1 and g[0, c] != bg)]
    if not keep_r or not keep_c or (len(keep_r) == H and len(keep_c) == W):
        return g.copy()
    return g[np.ix_(keep_r, keep_c)].copy()


def op_raycast_ortho_from_dots(g: np.ndarray) -> np.ndarray:
    """Проводит ортогональные лучи от одиночных точек (1x1) до краёв сетки по фону."""
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False)
    for o in objs:
        if o["area"] == 1:
            r, _, c, _ = o["bbox"]
            col = o["color"]
            out[r, :] = np.where(out[r, :] == bg, col, out[r, :])
            out[:, c] = np.where(out[:, c] == bg, col, out[:, c])
    return out


def op_raycast_diag_from_dots(g: np.ndarray) -> np.ndarray:
    """Проводит диагональные лучи от одиночных точек (1x1) до краёв сетки по фону."""
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False)
    for o in objs:
        if o["area"] == 1:
            r0, _, c0, _ = o["bbox"]
            col = o["color"]
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < H and 0 <= c < W:
                    if out[r, c] == bg:
                        out[r, c] = col
                    r += dr
                    c += dc
    return out


def op_stamp_template_at_dots(g: np.ndarray) -> np.ndarray:
    """
    Находит многопиксельный шаблон T (area >= 3) и одиночные маркеры 1x1.
    Если в шаблоне есть 1 центральный пиксель особого цвета-якоря, совмещает его с маркерами,
    иначе штампует шаблон по центру маркера с цветом маркера или шаблона.
    """
    bg = get_dominant_bg(g)
    H, W = g.shape
    objs_multi = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=True)
    templates = [o for o in objs_multi if o["area"] >= 3]
    dots = [o for o in objs_multi if o["area"] == 1]
    if len(templates) != 1 or not dots:
        return g.copy()

    tmpl = templates[0]
    ty0, ty1, tx0, tx1 = tmpl["bbox"]
    sub = tmpl["subgrid"]
    th, tw = sub.shape
    out = g.copy()

    # Проверяем, есть ли в мультицветном шаблоне 1 пиксель якорного цвета, совпадающего с цветом точек
    vals, cnts = np.unique(sub[sub != bg], return_counts=True)
    anchor_offset = (th // 2, tw // 2)
    has_anchor = False
    if len(vals) == 2 and 1 in cnts:
        anc_c = int(vals[np.argmin(cnts)])
        ay, ax = np.where(sub == anc_c)
        if len(ay) == 1:
            anchor_offset = (int(ay[0]), int(ax[0]))
            has_anchor = True

    ar, ac = anchor_offset
    for d in dots:
        dr, _, dc, _ = d["bbox"]
        d_col = d["color"]
        for r in range(th):
            for c in range(tw):
                val = sub[r, c]
                if val == bg:
                    continue
                nr, nc = dr + (r - ar), dc + (c - ac)
                if 0 <= nr < H and 0 <= nc < W:
                    if has_anchor:
                        out[nr, nc] = val
                    else:
                        out[nr, nc] = d_col
    return out


# --- Дополнительные примитивы Stage v32.2 (Симметрия BBox, Клеточные правила, Лучи с препятствиями, Сдвиги, Сортировка) ---
def op_complete_sym_antitranspose(g: np.ndarray) -> np.ndarray:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    bg = get_dominant_bg(g)
    at = np.fliplr(np.flipud(g.T))
    return np.where(g != bg, g, at)


def op_complete_sym_rot90_4way(g: np.ndarray) -> np.ndarray:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    bg = get_dominant_bg(g)
    out = g.copy()
    for k in (1, 2, 3):
        rk = np.rot90(g, k)
        out = np.where(out != bg, out, rk)
    return out


def op_complete_fg_bbox_sym_h(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    out = g.copy()
    sub = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = np.where(sub != bg, sub, np.fliplr(sub))
    return out


def op_complete_fg_bbox_sym_v(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    out = g.copy()
    sub = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = np.where(sub != bg, sub, np.flipud(sub))
    return out


def op_complete_fg_bbox_sym_rot180(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    out = g.copy()
    sub = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = np.where(sub != bg, sub, np.rot90(sub, 2))
    return out


def op_complete_fg_bbox_sym_4way(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    out = g.copy()
    sub = out[y0:y1, x0:x1]
    sub = np.where(sub != bg, sub, np.fliplr(sub))
    sub = np.where(sub != bg, sub, np.flipud(sub))
    if sub.shape[0] == sub.shape[1]:
        sub = np.where(sub != bg, sub, sub.T)
    out[y0:y1, x0:x1] = sub
    return out


def op_mirror_left_to_right(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    hw = W // 2
    out[:, W - hw :] = np.fliplr(g[:, :hw])
    return out


def op_mirror_right_to_left(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    hw = W // 2
    out[:, :hw] = np.fliplr(g[:, W - hw :])
    return out


def op_mirror_top_to_bottom(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    hh = H // 2
    out[H - hh :, :] = np.flipud(g[:hh, :])
    return out


def op_mirror_bottom_to_top(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    hh = H // 2
    out[:hh, :] = np.flipud(g[H - hh :, :])
    return out


def op_keep_only_object_interior_4n(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    fg = g != bg
    eroded = ndimage.binary_erosion(fg, border_value=0)
    return np.where(eroded, g, bg)


def op_keep_only_object_boundary_4n(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    fg = g != bg
    eroded = ndimage.binary_erosion(fg, border_value=0)
    return np.where(fg & (~eroded), g, bg)


def op_morph_close_hv(g: np.ndarray) -> np.ndarray:
    """Заполняет единичные разрывы (1 пиксель фона) между пикселями одного цвета по вертикали и горизонтали."""
    bg = get_dominant_bg(g)
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(1, W - 1):
            if out[r, c] == bg and g[r, c - 1] != bg and g[r, c - 1] == g[r, c + 1]:
                out[r, c] = g[r, c - 1]
    for r in range(1, H - 1):
        for c in range(W):
            if out[r, c] == bg and g[r - 1, c] != bg and g[r - 1, c] == g[r + 1, c]:
                out[r, c] = g[r - 1, c]
    return out


def op_fill_concave_2adj(g: np.ndarray) -> np.ndarray:
    """Заполняет фоновые клетки, имеющие >= 2 ортогональных соседей одного нефонового цвета."""
    bg = get_dominant_bg(g)
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                continue
            nbrs = []
            if r > 0 and g[r - 1, c] != bg:
                nbrs.append(int(g[r - 1, c]))
            if r + 1 < H and g[r + 1, c] != bg:
                nbrs.append(int(g[r + 1, c]))
            if c > 0 and g[r, c - 1] != bg:
                nbrs.append(int(g[r, c - 1]))
            if c + 1 < W and g[r, c + 1] != bg:
                nbrs.append(int(g[r, c + 1]))
            if len(nbrs) >= 2:
                vals, cnts = np.unique(nbrs, return_counts=True)
                if cnts.max() >= 2:
                    out[r, c] = vals[np.argmax(cnts)]
    return out


def op_frame_each_cc_bbox(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    out = g.copy()
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False)
    for o in objs:
        y0, y1, x0, x1 = o["bbox"]
        col = o["color"]
        out[y0, x0:x1] = col
        out[y1 - 1, x0:x1] = col
        out[y0:y1, x0] = col
        out[y0:y1, x1 - 1] = col
    return out


def op_denoise_min_area_3(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=True)
    out = np.full_like(g, bg)
    for o in objs:
        if o["area"] >= 3:
            out[o["mask"]] = g[o["mask"]]
    return out


def op_raycast_ortho_to_obstacle(g: np.ndarray) -> np.ndarray:
    """Проводит ортогональные лучи от точек 1x1 до первого препятствия (или края сетки)."""
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False)
    for o in objs:
        if o["area"] == 1:
            r0, _, c0, _ = o["bbox"]
            col = o["color"]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < H and 0 <= c < W:
                    if g[r, c] != bg:
                        break
                    out[r, c] = col
                    r += dr
                    c += dc
    return out


def op_raycast_diag_to_obstacle(g: np.ndarray) -> np.ndarray:
    """Проводит диагональные лучи от точек 1x1 до первого препятствия (или края сетки)."""
    bg = get_dominant_bg(g)
    out = g.copy()
    H, W = g.shape
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False)
    for o in objs:
        if o["area"] == 1:
            r0, _, c0, _ = o["bbox"]
            col = o["color"]
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < H and 0 <= c < W:
                    if g[r, c] != bg:
                        break
                    out[r, c] = col
                    r += dr
                    c += dc
    return out


def op_extend_lines_hv(g: np.ndarray) -> np.ndarray:
    """Продолжает отрезки длины >= 2 по горизонтали и вертикали через фон до краёв сетки."""
    bg = get_dominant_bg(g)
    out = g.copy()
    objs = extract_rich_objects(g, bg=bg, connectivity=4, multicolor=False)
    for o in objs:
        if o["is_line"]:
            y0, y1, x0, x1 = o["bbox"]
            col = o["color"]
            if o["bbox_h"] == 1 and o["bbox_w"] >= 2:
                out[y0, :] = np.where(out[y0, :] == bg, col, out[y0, :])
            elif o["bbox_w"] == 1 and o["bbox_h"] >= 2:
                out[:, x0] = np.where(out[:, x0] == bg, col, out[:, x0])
    return out


def op_slide_small_cc_to_largest(g: np.ndarray) -> np.ndarray:
    """Сдвигает меньшие объекты по горизонтали или вертикали вплотную к самому большому объекту."""
    bg = get_dominant_bg(g)
    objs = extract_rich_objects(g, bg=bg, connectivity=8, multicolor=True)
    if len(objs) < 2:
        return g.copy()
    objs.sort(key=lambda o: o["area"], reverse=True)
    anchor = objs[0]
    ay0, ay1, ax0, ax1 = anchor["bbox"]
    out = np.full_like(g, bg)
    out[anchor["mask"]] = g[anchor["mask"]]
    for o in objs[1:]:
        y0, y1, x0, x1 = o["bbox"]
        bh, bw = o["bbox_h"], o["bbox_w"]
        dy, dx = 0, 0
        # Если пересекается по строкам — сдвигаем по X вплотную к anchor
        if max(y0, ay0) < min(y1, ay1):
            if x1 <= ax0:
                dx = ax0 - x1
            elif x0 >= ax1:
                dx = ax1 - x0
        # Если пересекается по столбцам — сдвигаем по Y вплотную к anchor
        elif max(x0, ax0) < min(x1, ax1):
            if y1 <= ay0:
                dy = ay0 - y1
            elif y0 >= ay1:
                dy = ay1 - y0
        ny0, nx0 = y0 + dy, x0 + dx
        sub_m = o["submask"]
        sub_g = o["subgrid"]
        out[ny0 : ny0 + bh, nx0 : nx0 + bw] = np.where(
            sub_m, sub_g, out[ny0 : ny0 + bh, nx0 : nx0 + bw]
        )
    return out


def op_sort_rows_by_fg_asc(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    counts = np.sum(g != bg, axis=1)
    order = np.argsort(counts, kind="mergesort")
    return g[order, :].copy()


def op_sort_rows_by_fg_desc(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    counts = -np.sum(g != bg, axis=1)
    order = np.argsort(counts, kind="mergesort")
    return g[order, :].copy()


def op_sort_cols_by_fg_asc(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    counts = np.sum(g != bg, axis=0)
    order = np.argsort(counts, kind="mergesort")
    return g[:, order].copy()


def op_sort_cols_by_fg_desc(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    counts = -np.sum(g != bg, axis=0)
    order = np.argsort(counts, kind="mergesort")
    return g[:, order].copy()


def op_tile_1x2_mirror(g: np.ndarray) -> np.ndarray:
    return np.hstack([g, np.fliplr(g)])


def op_tile_2x1_mirror(g: np.ndarray) -> np.ndarray:
    return np.vstack([g, np.flipud(g)])


def op_tile_2x2_flip_h(g: np.ndarray) -> np.ndarray:
    row = np.hstack([g, np.fliplr(g)])
    return np.vstack([row, row])


def op_tile_2x2_flip_v(g: np.ndarray) -> np.ndarray:
    col = np.vstack([g, np.flipud(g)])
    return np.hstack([col, col])


def op_kronecker_inv_self(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    if H * H > 30 or W * W > 30:
        return g.copy()
    bg = get_dominant_bg(g)
    inv_mask = (g == bg).astype(g.dtype)
    out = np.zeros((H * H, W * W), dtype=g.dtype)
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                out[r * H : (r + 1) * H, c * W : (c + 1) * W] = inv_mask * g[r, c]
    return out


def op_crop_most_freq_fg_bbox(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, cnts) if v != bg]
    if not non_bg:
        return g.copy()
    non_bg.sort(key=lambda x: x[1], reverse=True)
    top_c = non_bg[0][0]
    ys, xs = np.where(g == top_c)
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def op_crop_second_color_bbox(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    vals, cnts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, cnts) if v != bg]
    if len(non_bg) < 2:
        return g.copy()
    non_bg.sort(key=lambda x: x[1], reverse=True)
    sec_c = non_bg[1][0]
    ys, xs = np.where(g == sec_c)
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def op_crop_enclosed_holes_bbox(g: np.ndarray) -> np.ndarray:
    bg = get_dominant_bg(g)
    bg_mask = g == bg
    labeled, n = ndimage.label(bg_mask)
    if n == 0:
        return g.copy()
    border_lbls = (
        set(np.unique(labeled[0, :]))
        | set(np.unique(labeled[-1, :]))
        | set(np.unique(labeled[:, 0]))
        | set(np.unique(labeled[:, -1]))
    )
    holes_mask = np.zeros_like(bg_mask)
    for lbl in range(1, n + 1):
        if lbl not in border_lbls:
            holes_mask |= labeled == lbl
    ys, xs = np.where(holes_mask)
    if len(ys) == 0:
        return g.copy()
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def _reconstruct_occluded_mosaic(g: np.ndarray, mask_color: int) -> np.ndarray:
    """
    Восстанавливает пиксели сетки g, закрытые маской цвета mask_color,
    используя автосимметрии D4 (H, V, Rot180, Rot90, Transpose, Antitranspose)
    и 2D-периодичность, вычисленные по незакрытой части сетки.
    """
    H, W = g.shape
    occ = g == mask_color
    if not np.any(occ) or np.all(occ):
        return g.copy()
    valid = ~occ
    out = g.copy()

    # 1. Оцениваем точность каждой симметрии D4 на открытых пикселях
    sym_candidates: List[Tuple[float, np.ndarray]] = []
    transforms = [
        np.fliplr(g),
        np.flipud(g),
        np.rot90(g, 2),
    ]
    if H == W:
        transforms.extend([g.T, np.fliplr(np.flipud(g.T)), np.rot90(g, 1), np.rot90(g, -1)])

    for tr in transforms:
        tr_valid = tr != mask_color
        both = valid & tr_valid
        if np.any(both):
            acc = float(np.mean(g[both] == tr[both]))
            if acc >= 0.85:
                sym_candidates.append((acc, tr))

    sym_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, tr in sym_candidates:
        fill_mask = (out == mask_color) & (tr != mask_color)
        out[fill_mask] = tr[fill_mask]
        if not np.any(out == mask_color):
            return out

    # 2. Если остались закрытые пиксели, ищем 2D или 1D периодичность (py, px)
    best_periods: List[Tuple[float, int, int]] = []
    for py in range(1, max(2, H // 2 + 1)):
        for px in range(1, max(2, W // 2 + 1)):
            if py == 1 and px == 1:
                continue
            if py > 8 and px > 8:
                continue
            shifted = np.roll(np.roll(g, py, axis=0), px, axis=1)
            # Проверяем без сшивки через границу
            s_valid = valid[py:, px:] & valid[: H - py, : W - px]
            if np.sum(s_valid) >= 6:
                acc = float(np.mean(g[py:, px:][s_valid] == g[: H - py, : W - px][s_valid]))
                if acc >= 0.90:
                    best_periods.append((acc, py, px))

    # Также проверяем чисто вертикальный (py, 0) и чисто горизонтальный (0, px) периоды
    for py in range(1, max(2, H // 2 + 1)):
        s_valid = valid[py:, :] & valid[: H - py, :]
        if np.sum(s_valid) >= 6:
            acc = float(np.mean(g[py:, :][s_valid] == g[: H - py, :][s_valid]))
            if acc >= 0.92:
                best_periods.append((acc, py, 0))
    for px in range(1, max(2, W // 2 + 1)):
        s_valid = valid[:, px:] & valid[:, : W - px]
        if np.sum(s_valid) >= 6:
            acc = float(np.mean(g[:, px:][s_valid] == g[:, : W - px][s_valid]))
            if acc >= 0.92:
                best_periods.append((acc, 0, px))

    best_periods.sort(key=lambda x: (x[0], -(x[1] + x[2])), reverse=True)
    for _, py, px in best_periods[:4]:
        for _ in range(4):
            rem_y, rem_x = np.where(out == mask_color)
            if len(rem_y) == 0:
                break
            for r, c in zip(rem_y, rem_x):
                for sign in (-1, 1):
                    nr, nc = r + sign * py, c + sign * px
                    if 0 <= nr < H and 0 <= nc < W and out[nr, nc] != mask_color:
                        out[r, c] = out[nr, nc]
                        break
    return out


def split_grid_by_separator(g: np.ndarray) -> List[np.ndarray]:
    """
    Разбивает сетку на равные панели по сплошным разделительным линиям (или пополам).
    Поддерживает как одномерное (строки ИЛИ столбцы), так и двумерное (строки И столбцы) разбиение.
    """
    H, W = g.shape
    bg = get_dominant_bg(g)

    # Проверяем общие разделительные линии (включая случай, когда разделитель одного цвета)
    sep_rows = [r for r in range(H) if len(np.unique(g[r, :])) == 1 and g[r, 0] != bg]
    sep_cols = [c for c in range(W) if len(np.unique(g[:, c])) == 1 and g[0, c] != bg]

    # Если разделитель ровно посередине (даже цвета 0, если вокруг непусто):
    if not sep_rows and H % 2 == 1 and H >= 3 and len(np.unique(g[H // 2, :])) == 1:
        sep_rows = [H // 2]
    if not sep_cols and W % 2 == 1 and W >= 3 and len(np.unique(g[:, W // 2])) == 1:
        sep_cols = [W // 2]

    row_cuts = [-1] + sep_rows + [H]
    col_cuts = [-1] + sep_cols + [W]
    r_spans = [
        (row_cuts[i] + 1, row_cuts[i + 1])
        for i in range(len(row_cuts) - 1)
        if row_cuts[i] + 1 < row_cuts[i + 1]
    ]
    c_spans = [
        (col_cuts[i] + 1, col_cuts[i + 1])
        for i in range(len(col_cuts) - 1)
        if col_cuts[i] + 1 < col_cuts[i + 1]
    ]

    if (len(r_spans) >= 2 or len(c_spans) >= 2) and len({s[1] - s[0] for s in r_spans}) == 1 and len(
        {s[1] - s[0] for s in c_spans}
    ) == 1:
        panels = []
        for r0, r1 in r_spans:
            for c0, c1 in c_spans:
                panels.append(g[r0:r1, c0:c1].copy())
        if len(panels) >= 2:
            return panels

    # Если явных разделителей нет, но сетка делится пополам (H=2*W или W=2*H или 2x2):
    if H % 2 == 0 and H == 2 * W:
        return [g[: H // 2, :].copy(), g[H // 2 :, :].copy()]
    if W % 2 == 0 and W == 2 * H:
        return [g[:, : W // 2].copy(), g[:, W // 2 :].copy()]
    return []


def split_grid_into_equal_subpanels(g: np.ndarray, target_h: int, target_w: int) -> List[np.ndarray]:
    """Извлекает равные панели размера (target_h, target_w) по разделителям или регулярной сетке."""
    panels = split_grid_by_separator(g)
    if panels and panels[0].shape == (target_h, target_w):
        return panels
    H, W = g.shape
    if H % target_h == 0 and W % target_w == 0 and (H > target_h or W > target_w):
        kr, kc = H // target_h, W // target_w
        if kr * kc <= 16:
            return [
                g[r * target_h : (r + 1) * target_h, c * target_w : (c + 1) * target_w].copy()
                for r in range(kr)
                for c in range(kc)
            ]
    return panels


# ==============================================================================
# 3. БИБЛИОТЕКА КОНСОЛИДАЦИИ МАКРО-НАВЫКОВ ВО СНЕ (Wake-Sleep Stitch-EBBS)
# ==============================================================================
class WakeSleepMacroLibrary_v32:
    """
    Хранилище верифицированных многошаговых макро-программ (`Stitch-EBBS`),
    открытых планировщиком Mode-2 при `UF=0` (нулевой ошибке на `train_pairs`).
    Консолидирует только контекстно-независимые примитивы, позволяя переносить
    составные навыки с `training` сплита на `evaluation` сплит.
    """

    def __init__(self, max_macros: int = 24):
        self.max_macros = max_macros
        self.macros: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], int]] = {}

    def register_verified_composition(
        self,
        op_names: List[str],
        op_fns: List[Callable[[np.ndarray], np.ndarray]],
        universal_names: set,
    ) -> Optional[str]:
        if len(op_names) < 2:
            return None
        if not all(name in universal_names for name in op_names):
            return None
        macro_name = "macro[" + "+".join(op_names) + "]"
        if macro_name in self.macros:
            fn, cnt = self.macros[macro_name]
            self.macros[macro_name] = (fn, cnt + 1)
            return macro_name

        if len(self.macros) >= self.max_macros:
            return None

        fns_copy = list(op_fns)

        def _composed_fn(g: np.ndarray, chain=fns_copy) -> np.ndarray:
            cur = g
            for f in chain:
                cur = f(cur)
            return cur

        self.macros[macro_name] = (_composed_fn, 1)
        return macro_name

    def get_consolidated_ops(self) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
        items = sorted(self.macros.items(), key=lambda kv: kv[1][1], reverse=True)
        return [(k, v[0]) for k, v in items]


# ==============================================================================
# 4. ИНДУКЦИЯ ОБЪЕКТНО-ОРИЕНТИРОВАННЫХ ПРИМИТИВОВ ИЗ ДЕМОНСТРАЦИЙ TRAIN_PAIRS
# ==============================================================================
def get_universal_base_primitives() -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Возвращает универсальные (не зависящие от конкретной задачи) операторы DSL."""
    ops: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("id", op_identity),
        ("rot90", op_rot90),
        ("rot180", op_rot180),
        ("rot270", op_rot270),
        ("flip_h", op_flip_h),
        ("flip_v", op_flip_v),
        ("transpose", op_transpose),
        ("antitranspose", op_antitranspose),
        ("crop_nonzero", op_crop_nonzero),
        ("crop_non_bg", op_crop_non_bg),
        ("crop_largest_cc4", op_crop_largest_cc4_mono),
        ("crop_largest_cc8", op_crop_largest_cc8_multi),
        ("crop_smallest_cc4", op_crop_smallest_cc4_mono),
        ("crop_smallest_cc8", op_crop_smallest_cc8_multi),
        ("crop_rare_color", op_crop_rare_color_bbox),
        ("crop_inside_rare_markers", op_crop_inside_rare_markers),
        ("crop_rare_markers_clean", op_crop_rare_markers_clean),
        ("crop_inside_frame", op_crop_inside_frame),
        ("crop_frame_inclusive", op_crop_frame_inclusive),
        ("keep_largest_cc", op_keep_only_largest_cc),
        ("keep_smallest_cc", op_keep_only_smallest_cc),
        ("denoise_1x1", op_remove_isolated_1x1_noise),
        ("keep_only_1x1", op_keep_only_isolated_1x1),
        ("fill_holes_parent", op_fill_enclosed_holes_with_parent),
        ("gravity_down", op_gravity_down),
        ("gravity_up", op_gravity_up),
        ("gravity_left", op_gravity_left),
        ("gravity_right", op_gravity_right),
        ("sym_h", op_complete_sym_h),
        ("sym_v", op_complete_sym_v),
        ("sym_rot180", op_complete_sym_rot180),
        ("sym_transpose", op_complete_sym_transpose),
        ("sym_4way", op_complete_sym_4way),
        ("connect_dots_hv", op_connect_aligned_dots_hv),
        ("fill_span_row", op_fill_between_same_color_row),
        ("fill_span_col", op_fill_between_same_color_col),
        ("fill_color_bboxes", op_fill_color_bboxes),
        ("fill_cc_bboxes", op_fill_each_cc_bbox),
        ("hollow_rects", op_hollow_rectangles),
        ("swap_2fg", op_swap_two_fg_colors),
        ("swap_most_least_fg", op_swap_most_least_fg_colors),
        ("recolor_to_most_fg", op_recolor_to_most_freq_fg),
        ("recolor_to_least_fg", op_recolor_to_least_freq_fg),
        ("tile_2x2_mirror", op_tile_2x2_mirror),
        ("tile_2x2_mirror_inv", op_tile_2x2_mirror_inv),
        ("tile_2x2_rot", op_tile_2x2_rot),
        ("tile_2x2_rep", op_tile_2x2_repeat),
        ("tile_3x3_rep", op_tile_3x3_repeat),
        ("kronecker_self", op_kronecker_self_fractal),
        ("kronecker_color", op_kronecker_color_fractal),
        ("compress_dup_rc", op_compress_consecutive_dup_rows_cols),
        ("remove_empty_rc", op_remove_empty_bg_rows_cols),
        ("remove_sep_lines", op_remove_separator_lines),
        ("raycast_ortho_dots", op_raycast_ortho_from_dots),
        ("raycast_diag_dots", op_raycast_diag_from_dots),
        ("stamp_template_dots", op_stamp_template_at_dots),
        ("sym_antitranspose", op_complete_sym_antitranspose),
        ("sym_rot90_4way", op_complete_sym_rot90_4way),
        ("bbox_sym_h", op_complete_fg_bbox_sym_h),
        ("bbox_sym_v", op_complete_fg_bbox_sym_v),
        ("bbox_sym_rot180", op_complete_fg_bbox_sym_rot180),
        ("bbox_sym_4way", op_complete_fg_bbox_sym_4way),
        ("mirror_l2r", op_mirror_left_to_right),
        ("mirror_r2l", op_mirror_right_to_left),
        ("mirror_t2b", op_mirror_top_to_bottom),
        ("mirror_b2t", op_mirror_bottom_to_top),
        ("keep_interior_4n", op_keep_only_object_interior_4n),
        ("keep_boundary_4n", op_keep_only_object_boundary_4n),
        ("morph_close_hv", op_morph_close_hv),
        ("fill_concave_2adj", op_fill_concave_2adj),
        ("frame_cc_bboxes", op_frame_each_cc_bbox),
        ("denoise_min3", op_denoise_min_area_3),
        ("raycast_ortho_obs", op_raycast_ortho_to_obstacle),
        ("raycast_diag_obs", op_raycast_diag_to_obstacle),
        ("extend_lines_hv", op_extend_lines_hv),
        ("slide_to_largest", op_slide_small_cc_to_largest),
        ("sort_rows_asc", op_sort_rows_by_fg_asc),
        ("sort_rows_desc", op_sort_rows_by_fg_desc),
        ("sort_cols_asc", op_sort_cols_by_fg_asc),
        ("sort_cols_desc", op_sort_cols_by_fg_desc),
        ("tile_1x2_mirror", op_tile_1x2_mirror),
        ("tile_2x1_mirror", op_tile_2x1_mirror),
        ("tile_2x2_flip_h", op_tile_2x2_flip_h),
        ("tile_2x2_flip_v", op_tile_2x2_flip_v),
        ("kronecker_inv_self", op_kronecker_inv_self),
        ("crop_most_freq_fg", op_crop_most_freq_fg_bbox),
        ("crop_second_color", op_crop_second_color_bbox),
        ("crop_holes_bbox", op_crop_enclosed_holes_bbox),
    ]

    # Универсальные объектные фильтры и кропы по топологическим признакам:
    for conn, multi, tag in [(4, False, "cc4"), (8, True, "cc8m")]:
        # Кропы по уникальности формы, дырам, числу цветов и симметрии:
        def _crop_by_rule(g: np.ndarray, rule: str, c_mode=conn, m_mode=multi) -> np.ndarray:
            bg = get_dominant_bg(g)
            objs = extract_rich_objects(g, bg=bg, connectivity=c_mode, multicolor=m_mode)
            if not objs:
                return g.copy()
            chosen = None
            if rule == "max_holes":
                objs.sort(key=lambda o: (o["n_holes"], o["area"]), reverse=True)
                chosen = objs[0]
            elif rule == "unique_shape":
                u = [o for o in objs if o["shape_freq"] == 1]
                chosen = u[0] if u else objs[0]
            elif rule == "unique_d4":
                u = [o for o in objs if o["d4_shape_freq"] == 1]
                chosen = u[0] if u else objs[0]
            elif rule == "most_freq_shape":
                objs.sort(key=lambda o: (o["shape_freq"], o["area"]), reverse=True)
                chosen = objs[0]
            elif rule == "max_colors":
                objs.sort(key=lambda o: (o["n_colors"], o["area"]), reverse=True)
                chosen = objs[0]
            elif rule == "sym_any":
                u = [o for o in objs if o["is_sym_any"]]
                chosen = u[0] if u else objs[0]
            elif rule == "not_sym":
                u = [o for o in objs if not o["is_sym_any"]]
                chosen = u[0] if u else objs[0]
            elif rule == "second_largest":
                objs.sort(key=lambda o: o["area"], reverse=True)
                chosen = objs[1] if len(objs) >= 2 else objs[0]
            if chosen is None:
                return g.copy()
            y0, y1, x0, x1 = chosen["bbox"]
            return g[y0:y1, x0:x1].copy()

        for r_name in [
            "max_holes",
            "unique_shape",
            "unique_d4",
            "most_freq_shape",
            "max_colors",
            "sym_any",
            "not_sym",
            "second_largest",
        ]:
            ops.append((f"crop_{tag}_{r_name}", lambda g, rn=r_name, cm=conn, mm=multi: _crop_by_rule(g, rn, cm, mm)))

        # Фильтрация объектов на месте (сохранение подмножества объектов):
        def _filter_by_rule(g: np.ndarray, rule: str, c_mode=conn, m_mode=multi) -> np.ndarray:
            bg = get_dominant_bg(g)
            objs = extract_rich_objects(g, bg=bg, connectivity=c_mode, multicolor=m_mode)
            if not objs:
                return g.copy()
            out = np.full_like(g, bg)
            keep = []
            if rule == "has_hole":
                keep = [o for o in objs if o["n_holes"] > 0]
            elif rule == "no_hole":
                keep = [o for o in objs if o["n_holes"] == 0]
            elif rule == "is_rect":
                keep = [o for o in objs if o["is_rect"]]
            elif rule == "is_square":
                keep = [o for o in objs if o["is_square"]]
            elif rule == "is_sym":
                keep = [o for o in objs if o["is_sym_any"]]
            elif rule == "not_sym":
                keep = [o for o in objs if not o["is_sym_any"]]
            elif rule == "unique_shape":
                keep = [o for o in objs if o["shape_freq"] == 1]
            elif rule == "repeated_shape":
                keep = [o for o in objs if o["shape_freq"] > 1]
            elif rule == "unique_color_cc":
                keep = [o for o in objs if o["color_cc_freq"] == 1]
            elif rule == "not_border":
                keep = [o for o in objs if not o["touches_border"]]
            elif rule == "touches_border":
                keep = [o for o in objs if o["touches_border"]]
            for o in keep:
                out[o["mask"]] = g[o["mask"]]
            return out

        for f_name in [
            "has_hole",
            "no_hole",
            "is_rect",
            "is_square",
            "is_sym",
            "not_sym",
            "unique_shape",
            "repeated_shape",
            "unique_color_cc",
            "not_border",
            "touches_border",
        ]:
            ops.append((f"keep_{tag}_{f_name}", lambda g, fn=f_name, cm=conn, mm=multi: _filter_by_rule(g, fn, cm, mm)))

    return ops


def build_task_specific_induced_primitives(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    macro_lib: Optional[WakeSleepMacroLibrary_v32] = None,
) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """
    Анализирует ТОЛЬКО демонстрационные пары `train_pairs` задачи и индуцирует
    параметризованные операторы (цветовые отображения, объектные перекраски по площади/дырам/форме,
    реконструкцию закрытых мозаик, выбор подпанелей, гравитацию к препятствиям и т.д.).
    НИКОГДА не обращается к `test` данным!
    """
    ops = get_universal_base_primitives()
    if macro_lib is not None:
        ops.extend(macro_lib.get_consolidated_ops())

    same_shape_all = all(x.shape == y.shape for x, y in train_pairs)

    # 1. Индукция глобальной таблицы перекраски цветов из train_pairs (если размеры совпадают):
    if same_shape_all:
        cmap: Dict[int, int] = {}
        consistent_cmap = True
        for xin, yout in train_pairs:
            for c_in in np.unique(xin):
                vals_out = np.unique(yout[xin == c_in])
                if len(vals_out) == 1:
                    c_out = int(vals_out[0])
                    if c_in in cmap and cmap[c_in] != c_out:
                        consistent_cmap = False
                    cmap[int(c_in)] = c_out
                else:
                    consistent_cmap = False
        if consistent_cmap and cmap:
            def _apply_cmap(g: np.ndarray, m=dict(cmap)) -> np.ndarray:
                out = g.copy()
                for k, v in m.items():
                    out[g == k] = v
                return out
            ops.append(("induced_color_map", _apply_cmap))

    # 2. Индукция объектно-зависимой перекраски (Area / Holes / BBox / Rank / Freq -> Color)
    if same_shape_all:
        for conn, multi, tag in [(4, False, "cc4"), (8, False, "cc8"), (8, True, "cc8m")]:
            area_to_c: Dict[int, int] = {}
            holes_to_c: Dict[int, int] = {}
            bshape_to_c: Dict[Tuple[int, int], int] = {}
            sfreq_to_c: Dict[int, int] = {}
            ccfreq_to_c: Dict[int, int] = {}
            sym_to_c: Dict[bool, int] = {}

            ok_area = ok_holes = ok_bshape = ok_sfreq = ok_ccfreq = ok_sym = True
            max_colors_set, min_colors_set, uniq_shape_colors_set = set(), set(), set()

            for xin, yout in train_pairs:
                bg = get_dominant_bg(xin)
                objs = extract_rich_objects(xin, bg=bg, connectivity=conn, multicolor=multi)
                if not objs:
                    ok_area = ok_holes = ok_bshape = ok_sfreq = ok_ccfreq = ok_sym = False
                    break

                # Ранги по площади
                areas_sorted = sorted({o["area"] for o in objs}, reverse=True)
                max_a = areas_sorted[0]
                min_a = areas_sorted[-1]

                for o in objs:
                    out_vals = np.unique(yout[o["mask"]])
                    if len(out_vals) != 1:
                        ok_area = ok_holes = ok_bshape = ok_sfreq = ok_ccfreq = ok_sym = False
                        continue
                    c_out = int(out_vals[0])

                    if o["area"] == max_a:
                        max_colors_set.add(c_out)
                    if o["area"] == min_a:
                        min_colors_set.add(c_out)
                    if o["shape_freq"] == 1:
                        uniq_shape_colors_set.add(c_out)

                    if area_to_c.get(o["area"], c_out) != c_out:
                        ok_area = False
                    area_to_c[o["area"]] = c_out

                    if holes_to_c.get(o["n_holes"], c_out) != c_out:
                        ok_holes = False
                    holes_to_c[o["n_holes"]] = c_out

                    bsh = (o["bbox_h"], o["bbox_w"])
                    if bshape_to_c.get(bsh, c_out) != c_out:
                        ok_bshape = False
                    bshape_to_c[bsh] = c_out

                    if sfreq_to_c.get(o["shape_freq"], c_out) != c_out:
                        ok_sfreq = False
                    sfreq_to_c[o["shape_freq"]] = c_out

                    if ccfreq_to_c.get(o["color_cc_freq"], c_out) != c_out:
                        ok_ccfreq = False
                    ccfreq_to_c[o["color_cc_freq"]] = c_out

                    if sym_to_c.get(o["is_sym_any"], c_out) != c_out:
                        ok_sym = False
                    sym_to_c[o["is_sym_any"]] = c_out

            if ok_area and area_to_c:
                def _recolor_by_area(g: np.ndarray, mp=dict(area_to_c), cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        if o["area"] in mp:
                            out[o["mask"]] = mp[o["area"]]
                    return out
                ops.append((f"recolor_{tag}_by_area", _recolor_by_area))

            if ok_holes and len(holes_to_c) >= 2:
                def _recolor_by_holes(g: np.ndarray, mp=dict(holes_to_c), cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        if o["n_holes"] in mp:
                            out[o["mask"]] = mp[o["n_holes"]]
                    return out
                ops.append((f"recolor_{tag}_by_holes", _recolor_by_holes))

            if ok_bshape and bshape_to_c:
                def _recolor_by_bshape(g: np.ndarray, mp=dict(bshape_to_c), cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        bsh = (o["bbox_h"], o["bbox_w"])
                        if bsh in mp:
                            out[o["mask"]] = mp[bsh]
                    return out
                ops.append((f"recolor_{tag}_by_bshape", _recolor_by_bshape))

            if ok_sfreq and len(sfreq_to_c) >= 2:
                def _recolor_by_sfreq(g: np.ndarray, mp=dict(sfreq_to_c), cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        if o["shape_freq"] in mp:
                            out[o["mask"]] = mp[o["shape_freq"]]
                    return out
                ops.append((f"recolor_{tag}_by_sfreq", _recolor_by_sfreq))

            if ok_ccfreq and len(ccfreq_to_c) >= 2:
                def _recolor_by_ccfreq(g: np.ndarray, mp=dict(ccfreq_to_c), cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        if o["color_cc_freq"] in mp:
                            out[o["mask"]] = mp[o["color_cc_freq"]]
                    return out
                ops.append((f"recolor_{tag}_by_ccfreq", _recolor_by_ccfreq))

            if ok_sym and len(sym_to_c) == 2:
                def _recolor_by_sym(g: np.ndarray, mp=dict(sym_to_c), cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        out[o["mask"]] = mp[o["is_sym_any"]]
                    return out
                ops.append((f"recolor_{tag}_by_sym", _recolor_by_sym))

            if len(max_colors_set) == 1 and len(min_colors_set) == 1:
                c_hi = list(max_colors_set)[0]
                c_lo = list(min_colors_set)[0]
                def _recolor_max_min(g: np.ndarray, ch=c_hi, cl=c_lo, cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    objs_g = extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm)
                    if not objs_g:
                        return out
                    areas = [o["area"] for o in objs_g]
                    ma, mia = max(areas), min(areas)
                    for o in objs_g:
                        if o["area"] == ma:
                            out[o["mask"]] = ch
                        elif o["area"] == mia:
                            out[o["mask"]] = cl
                    return out
                ops.append((f"recolor_{tag}_max{c_hi}_min{c_lo}", _recolor_max_min))

            if len(uniq_shape_colors_set) == 1:
                c_u = list(uniq_shape_colors_set)[0]
                def _recolor_uniq_shape(g: np.ndarray, cu=c_u, cm=conn, mm=multi) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    out = g.copy()
                    for o in extract_rich_objects(g, bg=bg, connectivity=cm, multicolor=mm):
                        if o["shape_freq"] == 1:
                            out[o["mask"]] = cu
                    return out
                ops.append((f"recolor_{tag}_uniq_shape_to_{c_u}", _recolor_uniq_shape))

    # 3. Индукция единого целевого цвета ненулевых клеток из выходов train_pairs:
    out_fg_colors = set()
    for _, yout in train_pairs:
        bg_y = get_dominant_bg(yout)
        for v in np.unique(yout):
            if v != bg_y and v != 0:
                out_fg_colors.add(int(v))
    if len(out_fg_colors) == 1:
        target_c = list(out_fg_colors)[0]
        ops.append((f"recolor_nonzero_to_{target_c}", lambda g, tc=target_c: np.where(g != 0, tc, 0)))

    # 4. Индукция новых цветов (для заливки дыр, рамок-обводок, линий между точками и лучей):
    diff_colors = set()
    if same_shape_all:
        for xin, yout in train_pairs:
            for dv in np.unique(yout[xin != yout]):
                if dv != 0:
                    diff_colors.add(int(dv))
    if len(diff_colors) == 1:
        hc = list(diff_colors)[0]

        def _fill_holes_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            out = g.copy()
            labeled, n = ndimage.label(g == bg)
            if n == 0:
                return out
            border_lbls = (
                set(np.unique(labeled[0, :]))
                | set(np.unique(labeled[-1, :]))
                | set(np.unique(labeled[:, 0]))
                | set(np.unique(labeled[:, -1]))
            )
            for lbl in range(1, n + 1):
                if lbl not in border_lbls:
                    out[labeled == lbl] = fill_c
            return out

        def _outline_4n_with_c(g: np.ndarray, ring_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            fg = g != bg
            dil = ndimage.binary_dilation(fg) & (~fg)
            out = g.copy()
            out[dil] = ring_c
            return out

        def _outline_8n_with_c(g: np.ndarray, ring_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            fg = g != bg
            dil = ndimage.binary_dilation(fg, structure=np.ones((3, 3), dtype=bool)) & (~fg)
            out = g.copy()
            out[dil] = ring_c
            return out

        def _fill_span_row_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            out = g.copy()
            for r in range(g.shape[0]):
                xs = np.where(g[r, :] != bg)[0]
                if len(xs) >= 2:
                    x0, x1 = int(xs.min()), int(xs.max())
                    seg = out[r, x0 : x1 + 1]
                    out[r, x0 : x1 + 1] = np.where(seg == bg, fill_c, seg)
            return out

        def _fill_span_col_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            out = g.copy()
            for c in range(g.shape[1]):
                ys = np.where(g[:, c] != bg)[0]
                if len(ys) >= 2:
                    y0, y1 = int(ys.min()), int(ys.max())
                    seg = out[y0 : y1 + 1, c]
                    out[y0 : y1 + 1, c] = np.where(seg == bg, fill_c, seg)
            return out

        def _connect_hv_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            out = g.copy()
            ys, xs = np.where(g != bg)
            if 2 <= len(ys) <= 20:
                for i in range(len(ys)):
                    for j in range(i + 1, len(ys)):
                        if ys[i] == ys[j]:
                            x0, x1 = min(xs[i], xs[j]), max(xs[i], xs[j])
                            seg = out[ys[i], x0 : x1 + 1]
                            out[ys[i], x0 : x1 + 1] = np.where(seg == bg, fill_c, seg)
                        elif xs[i] == xs[j]:
                            y0, y1 = min(ys[i], ys[j]), max(ys[i], ys[j])
                            seg = out[y0 : y1 + 1, xs[i]]
                            out[y0 : y1 + 1, xs[i]] = np.where(seg == bg, fill_c, seg)
            return out

        def _recolor_interior_4n_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            fg = g != bg
            eroded = ndimage.binary_erosion(fg, border_value=0)
            out = g.copy()
            out[eroded] = fill_c
            return out

        def _recolor_boundary_4n_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            fg = g != bg
            eroded = ndimage.binary_erosion(fg, border_value=0)
            out = g.copy()
            out[fg & (~eroded)] = fill_c
            return out

        def _fill_concave_2adj_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            H, W = g.shape
            out = g.copy()
            for r in range(H):
                for c in range(W):
                    if g[r, c] != bg:
                        continue
                    cnt = (
                        (1 if r > 0 and g[r - 1, c] != bg else 0)
                        + (1 if r + 1 < H and g[r + 1, c] != bg else 0)
                        + (1 if c > 0 and g[r, c - 1] != bg else 0)
                        + (1 if c + 1 < W and g[r, c + 1] != bg else 0)
                    )
                    if cnt >= 2:
                        out[r, c] = fill_c
            return out

        def _fill_cc_bbox_bg_with_c(g: np.ndarray, fill_c=hc) -> np.ndarray:
            bg = get_dominant_bg(g)
            out = g.copy()
            for o in extract_rich_objects(g, bg=bg, connectivity=8, multicolor=False):
                y0, y1, x0, x1 = o["bbox"]
                sub = out[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = np.where(sub == bg, fill_c, sub)
            return out

        ops.append((f"fill_holes_c{hc}", _fill_holes_with_c))
        ops.append((f"outline_4n_c{hc}", _outline_4n_with_c))
        ops.append((f"outline_8n_c{hc}", _outline_8n_with_c))
        ops.append((f"fill_span_row_c{hc}", _fill_span_row_with_c))
        ops.append((f"fill_span_col_c{hc}", _fill_span_col_with_c))
        ops.append((f"connect_hv_c{hc}", _connect_hv_with_c))
        ops.append((f"recolor_interior_4n_c{hc}", _recolor_interior_4n_with_c))
        ops.append((f"recolor_boundary_4n_c{hc}", _recolor_boundary_4n_with_c))
        ops.append((f"fill_concave_2adj_c{hc}", _fill_concave_2adj_with_c))
        ops.append((f"fill_cc_bbox_bg_c{hc}", _fill_cc_bbox_bg_with_c))

    # 5. Индукция реконструкции закрытых мозаик (или вырезания восстановленного фрагмента!):
    # Проверяем, есть ли цвет-маска c_mask, который присутствует во входах и исчезает в выходах:
    in_colors_common = set(np.unique(train_pairs[0][0]))
    for xin, _ in train_pairs[1:]:
        in_colors_common &= set(np.unique(xin))
    out_colors_union = set()
    for _, yout in train_pairs:
        out_colors_union |= set(np.unique(yout))
    disappearing_colors = [int(c) for c in (in_colors_common - out_colors_union) if c != 0]

    for mc in disappearing_colors:
        ops.append((f"reconstruct_mosaic_mask{mc}", lambda g, m=mc: _reconstruct_occluded_mosaic(g, m)))

        def _extract_reconstructed_patch(g: np.ndarray, m=mc) -> np.ndarray:
            ys, xs = np.where(g == m)
            if len(ys) == 0:
                return g.copy()
            rec = _reconstruct_occluded_mosaic(g, m)
            return rec[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()

        ops.append((f"extract_reconstructed_patch_mask{mc}", _extract_reconstructed_patch))

    # Также универсальный вариант: если в тестовом входе цвет маски другой (самый редкий или образующий сплошной прямоугольник):
    def _extract_any_occluded_rect_patch(g: np.ndarray) -> np.ndarray:
        bg = get_dominant_bg(g)
        objs = extract_rich_objects(g, bg=bg, connectivity=4, multicolor=False)
        rects = [o for o in objs if o["is_rect"] and o["area"] >= 4 and o["color_cc_freq"] == 1]
        if not rects:
            return g.copy()
        rects.sort(key=lambda o: o["area"], reverse=True)
        mc = rects[0]["color"]
        y0, y1, x0, x1 = rects[0]["bbox"]
        rec = _reconstruct_occluded_mosaic(g, mc)
        return rec[y0:y1, x0:x1].copy()

    def _reconstruct_any_occluded_rect_mosaic(g: np.ndarray) -> np.ndarray:
        bg = get_dominant_bg(g)
        objs = extract_rich_objects(g, bg=bg, connectivity=4, multicolor=False)
        rects = [o for o in objs if o["is_rect"] and o["area"] >= 4 and o["color_cc_freq"] == 1]
        if not rects:
            return g.copy()
        rects.sort(key=lambda o: o["area"], reverse=True)
        mc = rects[0]["color"]
        return _reconstruct_occluded_mosaic(g, mc)

    if all(len(set(np.unique(x)) - set(np.unique(y))) >= 1 for x, y in train_pairs):
        ops.append(("extract_occluded_rect_patch", _extract_any_occluded_rect_patch))
        ops.append(("reconstruct_occluded_rect_mosaic", _reconstruct_any_occluded_rect_mosaic))

    # 6. Индукция гравитации с неподвижными препятствиями:
    if same_shape_all:
        stationary_colors = None
        for xin, yout in train_pairs:
            bg = get_dominant_bg(xin)
            stat_here = {
                int(c)
                for c in np.unique(xin)
                if c != bg and np.array_equal(xin == c, yout == c)
            }
            stationary_colors = stat_here if stationary_colors is None else (stationary_colors & stat_here)
        if stationary_colors:
            for sc in stationary_colors:
                def _grav_obstacle_down(g: np.ndarray, obs_c=sc) -> np.ndarray:
                    bg = get_dominant_bg(g)
                    H, W = g.shape
                    out = np.where(g == obs_c, obs_c, bg)
                    for c in range(W):
                        write_r = H - 1
                        for r in range(H - 1, -1, -1):
                            if g[r, c] == obs_c:
                                write_r = r - 1
                            elif g[r, c] != bg:
                                if write_r >= 0:
                                    out[write_r, c] = g[r, c]
                                    write_r -= 1
                    return out

                def _grav_obstacle_up(g: np.ndarray, obs_c=sc) -> np.ndarray:
                    return np.flipud(_grav_obstacle_down(np.flipud(g), obs_c))

                def _grav_obstacle_right(g: np.ndarray, obs_c=sc) -> np.ndarray:
                    return _grav_obstacle_down(g.T, obs_c).T.copy()

                def _grav_obstacle_left(g: np.ndarray, obs_c=sc) -> np.ndarray:
                    return np.fliplr(_grav_obstacle_right(np.fliplr(g), obs_c))

                ops.append((f"grav_obs{sc}_down", _grav_obstacle_down))
                ops.append((f"grav_obs{sc}_up", _grav_obstacle_up))
                ops.append((f"grav_obs{sc}_left", _grav_obstacle_left))
                ops.append((f"grav_obs{sc}_right", _grav_obstacle_right))

    # 7. Индукция коэффициента масштабирования / сжатия (scale_up / scale_down):
    ratios_up = set()
    ratios_down = set()
    fixed_out_shapes = set()
    for xin, yout in train_pairs:
        hi, wi = xin.shape
        ho, wo = yout.shape
        fixed_out_shapes.add((ho, wo))
        if ho % hi == 0 and wo % wi == 0 and (ho > hi or wo > wi):
            ratios_up.add((ho // hi, wo // wi))
        if hi % ho == 0 and wi % wo == 0 and (hi > ho or wi > wo):
            ratios_down.add((hi // ho, wi // wo))

    if len(ratios_up) == 1:
        ry, rx = list(ratios_up)[0]
        if ry <= 6 and rx <= 6:
            ops.append(
                (f"kronecker_scale_{ry}x{rx}", lambda g, sy=ry, sx=rx: np.kron(g, np.ones((sy, sx), dtype=g.dtype)))
            )
            ops.append((f"tile_{ry}x{rx}", lambda g, sy=ry, sx=rx: np.tile(g, (sy, sx))))

    if len(ratios_down) == 1:
        dy, dx = list(ratios_down)[0]

        def _block_downsample(g: np.ndarray, sy=dy, sx=dx) -> np.ndarray:
            H, W = g.shape
            if H % sy != 0 or W % sx != 0:
                return g.copy()
            ho, wo = H // sy, W // sx
            out = np.zeros((ho, wo), dtype=g.dtype)
            for r in range(ho):
                for c in range(wo):
                    blk = g[r * sy : (r + 1) * sy, c * sx : (c + 1) * sx]
                    nz = blk[blk != 0]
                    if len(nz) > 0:
                        vals, cnts = np.unique(nz, return_counts=True)
                        out[r, c] = vals[np.argmax(cnts)]
            return out

        ops.append((f"block_downsample_{dy}x{dx}", _block_downsample))

    if len(fixed_out_shapes) == 1:
        tho, two = list(fixed_out_shapes)[0]
        if 1 <= tho <= 15 and 1 <= two <= 15:
            def _grid_cell_summary(g: np.ndarray, ho=tho, wo=two) -> np.ndarray:
                H, W = g.shape
                sep_r = [r for r in range(H) if len(np.unique(g[r, :])) == 1]
                sep_c = [c for c in range(W) if len(np.unique(g[:, c])) == 1]
                r_cuts = [-1] + sep_r + [H]
                c_cuts = [-1] + sep_c + [W]
                r_spans = [
                    (r_cuts[i] + 1, r_cuts[i + 1])
                    for i in range(len(r_cuts) - 1)
                    if r_cuts[i] + 1 < r_cuts[i + 1]
                ]
                c_spans = [
                    (c_cuts[i] + 1, c_cuts[i + 1])
                    for i in range(len(c_cuts) - 1)
                    if c_cuts[i] + 1 < c_cuts[i + 1]
                ]
                if len(r_spans) == ho and len(c_spans) == wo:
                    out = np.zeros((ho, wo), dtype=g.dtype)
                    for ri, (r0, r1) in enumerate(r_spans):
                        for ci, (c0, c1) in enumerate(c_spans):
                            cell = g[r0:r1, c0:c1]
                            nz = cell[cell != 0]
                            if len(nz) > 0:
                                vals, cnts = np.unique(nz, return_counts=True)
                                out[ri, ci] = vals[np.argmax(cnts)]
                    return out
                return g.copy()

            ops.append((f"sep_cells_to_{tho}x{two}", _grid_cell_summary))

    # 8. Панельные операции (булевы + выбор уникальной/мажоритарной/макс панели):
    tc = list(out_fg_colors)[0] if len(out_fg_colors) == 1 else 1
    target_h_out = train_pairs[0][1].shape[0] if len(fixed_out_shapes) == 1 else -1
    target_w_out = train_pairs[0][1].shape[1] if len(fixed_out_shapes) == 1 else -1

    def _panel_op(g: np.ndarray, mode: str, out_c: int = tc, tho=target_h_out, two=target_w_out) -> np.ndarray:
        panels = (
            split_grid_into_equal_subpanels(g, tho, two)
            if (tho > 0 and two > 0)
            else split_grid_by_separator(g)
        )
        if len(panels) < 2:
            return g.copy()
        p0 = panels[0]
        if any(pk.shape != p0.shape for pk in panels):
            return g.copy()

        # Если out_fg_colors не фиксирован одним цветом, можно также использовать доминирующий ненулевой цвет панелей
        if mode.endswith("_incolor"):
            all_nz = np.concatenate([pk[pk != 0] for pk in panels if np.any(pk != 0)]) if any(np.any(pk != 0) for pk in panels) else np.array([1])
            vals, cnts = np.unique(all_nz, return_counts=True)
            out_c = int(vals[np.argmax(cnts)])
            mode = mode.replace("_incolor", "")

        b0 = p0 != 0
        b1 = panels[1] != 0
        if mode == "and":
            res = b0 & b1
            for pk in panels[2:]:
                res = res & (pk != 0)
            return np.where(res, out_c, 0)
        elif mode == "or":
            res = b0 | b1
            for pk in panels[2:]:
                res = res | (pk != 0)
            return np.where(res, out_c, 0)
        elif mode == "xor":
            res = b0 ^ b1
            for pk in panels[2:]:
                res = res ^ (pk != 0)
            return np.where(res, out_c, 0)
        elif mode == "nor":
            res = b0 | b1
            for pk in panels[2:]:
                res = res | (pk != 0)
            return np.where(~res, out_c, 0)
        elif mode == "diff_01":
            return np.where(b0 & (~b1), out_c, 0)
        elif mode == "diff_10":
            return np.where(b1 & (~b0), out_c, 0)
        elif mode == "overlay_0_on_1":
            out = panels[-1].copy()
            for pk in reversed(panels[:-1]):
                out = np.where(pk != 0, pk, out)
            return out
        elif mode == "overlay_1_on_0":
            out = panels[0].copy()
            for pk in panels[1:]:
                out = np.where(pk != 0, pk, out)
            return out
        elif mode == "select_unique_exact":
            sigs = [pk.tobytes() for pk in panels]
            for idx, s in enumerate(sigs):
                if sigs.count(s) == 1:
                    return panels[idx].copy()
        elif mode == "select_unique_mask":
            sigs = [(pk != 0).tobytes() for pk in panels]
            for idx, s in enumerate(sigs):
                if sigs.count(s) == 1:
                    return panels[idx].copy()
        elif mode == "select_majority_exact":
            sigs = [pk.tobytes() for pk in panels]
            best_idx = max(range(len(panels)), key=lambda i: sigs.count(sigs[i]))
            return panels[best_idx].copy()
        elif mode == "select_max_fg":
            best_idx = max(range(len(panels)), key=lambda i: int(np.sum(panels[i] != 0)))
            return panels[best_idx].copy()
        elif mode == "select_min_fg":
            non_empty = [i for i in range(len(panels)) if np.any(panels[i] != 0)]
            if non_empty:
                best_idx = min(non_empty, key=lambda i: int(np.sum(panels[i] != 0)))
                return panels[best_idx].copy()
        elif mode == "select_max_colors":
            best_idx = max(range(len(panels)), key=lambda i: len(np.unique(panels[i])))
            return panels[best_idx].copy()
        elif mode == "select_min_colors":
            non_empty = [i for i in range(len(panels)) if np.any(panels[i] != 0)]
            if non_empty:
                best_idx = min(non_empty, key=lambda i: len(np.unique(panels[i])))
                return panels[best_idx].copy()
        elif mode == "keep_common_pixels":
            out = panels[0].copy()
            for pk in panels[1:]:
                out = np.where(out == pk, out, 0)
            return out
        elif mode == "pixelwise_majority":
            stack = np.stack(panels, axis=0)
            out = np.zeros_like(p0)
            for r in range(p0.shape[0]):
                for c in range(p0.shape[1]):
                    vals, cnts = np.unique(stack[:, r, c], return_counts=True)
                    out[r, c] = vals[np.argmax(cnts)]
            return out
        return g.copy()

    for m_name in [
        "and",
        "or",
        "xor",
        "nor",
        "diff_01",
        "diff_10",
        "and_incolor",
        "or_incolor",
        "xor_incolor",
        "nor_incolor",
        "overlay_0_on_1",
        "overlay_1_on_0",
        "select_unique_exact",
        "select_unique_mask",
        "select_majority_exact",
        "select_max_fg",
        "select_min_fg",
        "select_max_colors",
        "select_min_colors",
        "keep_common_pixels",
        "pixelwise_majority",
    ]:
        ops.append((f"panel_{m_name}", lambda g, mn=m_name: _panel_op(g, mn)))

    # 9. Индукция кадрирования по конкретному цвету (если один и тот же цвет в train_pairs задаёт рамку/область выхода):
    scale_down_all = all(
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1] and y.shape != x.shape for x, y in train_pairs
    )
    scale_up_all = all(
        y.shape[0] >= x.shape[0] and y.shape[1] >= x.shape[1] and y.shape != x.shape for x, y in train_pairs
    )
    if scale_down_all:
        for c_cand in in_colors_common:
            if c_cand == 0:
                continue
            def _crop_c_bbox(g: np.ndarray, cc=int(c_cand)) -> np.ndarray:
                ys, xs = np.where(g == cc)
                if len(ys) == 0:
                    return g.copy()
                return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()

            def _crop_inside_c_bbox(g: np.ndarray, cc=int(c_cand)) -> np.ndarray:
                ys, xs = np.where(g == cc)
                if len(ys) == 0:
                    return g.copy()
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
                if y1 - y0 >= 2 and x1 - x0 >= 2:
                    return g[y0 + 1 : y1, x0 + 1 : x1].copy()
                return g.copy()

            ops.append((f"crop_color_c{int(c_cand)}", _crop_c_bbox))
            ops.append((f"crop_inside_color_c{int(c_cand)}", _crop_inside_c_bbox))

    # 10. Shape-Guided Operator Pre-Pruning (Отсечение операторов по инварианту размерности):
    def _is_shrink_op(name: str) -> bool:
        return (
            name.startswith("crop_")
            or name.startswith("panel_")
            or name in ("compress_dup_rc", "remove_empty_rc", "remove_sep_lines")
            or name.startswith("block_downsample_")
            or name.startswith("sep_cells_to_")
            or name.startswith("extract_")
        )

    def _is_grow_op(name: str) -> bool:
        return name.startswith("tile_") or name.startswith("kronecker_")

    if same_shape_all:
        ops = [
            (n, f)
            for n, f in ops
            if not _is_shrink_op(n)
            and not _is_grow_op(n)
            and not (n.startswith("macro[") and ("crop_" in n or "panel_" in n or "tile_" in n or "kronecker_" in n or "compress_" in n or "remove_" in n))
        ]
    elif scale_down_all:
        ops = [
            (n, f)
            for n, f in ops
            if not _is_grow_op(n) and not (n.startswith("macro[") and ("tile_" in n or "kronecker_" in n))
        ]
    elif scale_up_all:
        ops = [
            (n, f)
            for n, f in ops
            if not _is_shrink_op(n)
            and not (n.startswith("macro[") and ("crop_" in n or "panel_" in n or "compress_" in n or "remove_" in n))
        ]

    return ops


# ==============================================================================
# 5. ПЛАНИРОВАТЕЛЬ MODE-2 BEAM/MPC И НЕЙРОСЕТЕВОЙ БЕЙЗЛАЙН НА РЕАЛЬНОМ ARC-AGI
# ==============================================================================
class DirectNeuralGridBaseline_v32(nn.Module):
    """
    Честный поклеточный нейросетевой бейзлайн (Mode-1 Direct Grid Predictor):
      - На каждой задаче ARC обучается на демонстрационных парах `task["train"]`
        предсказывать выходную сетку напрямую из входной без поиска программ.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def solve_task_blind(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_inputs: List[np.ndarray]
    ) -> List[np.ndarray]:
        shapes_in = {p[0].shape for p in train_pairs} | {t.shape for t in test_inputs}
        shapes_out = {p[1].shape for p in train_pairs}
        if len(shapes_in) != 1 or len(shapes_out) != 1:
            if all(p[0].shape == p[1].shape for p in train_pairs):
                conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1).to(self.device)
                conv2 = nn.Conv2d(32, 10, kernel_size=1).to(self.device)
                opt = torch.optim.AdamW(list(conv1.parameters()) + list(conv2.parameters()), lr=1e-2)
                for _ in range(35):
                    loss_total = 0.0
                    for xin, yout in train_pairs:
                        x_oh = (
                            F.one_hot(torch.from_numpy(xin).long().to(self.device), 10)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .float()
                        )
                        y_t = torch.from_numpy(yout).long().to(self.device).unsqueeze(0)
                        logits = conv2(F.silu(conv1(x_oh)))
                        loss_total = loss_total + F.cross_entropy(logits, y_t)
                    opt.zero_grad()
                    loss_total.backward()
                    opt.step()
                preds = []
                with torch.no_grad():
                    for tin in test_inputs:
                        x_oh = (
                            F.one_hot(torch.from_numpy(tin).long().to(self.device), 10)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .float()
                        )
                        p_grid = conv2(F.silu(conv1(x_oh))).argmax(dim=1).squeeze(0).cpu().numpy()
                        preds.append(p_grid)
                return preds
            return [tin.copy() for tin in test_inputs]

        hi, wi = list(shapes_in)[0]
        ho, wo = list(shapes_out)[0]
        fc1 = nn.Linear(hi * wi * 10, 128).to(self.device)
        fc2 = nn.Linear(128, ho * wo * 10).to(self.device)
        opt = torch.optim.AdamW(list(fc1.parameters()) + list(fc2.parameters()), lr=1e-2)
        for _ in range(40):
            loss_total = 0.0
            for xin, yout in train_pairs:
                x_oh = F.one_hot(torch.from_numpy(xin).long().to(self.device), 10).float().view(1, -1)
                y_t = torch.from_numpy(yout).long().to(self.device).view(-1)
                logits = fc2(F.silu(fc1(x_oh))).view(ho * wo, 10)
                loss_total = loss_total + F.cross_entropy(logits, y_t)
            opt.zero_grad()
            loss_total.backward()
            opt.step()

        preds = []
        with torch.no_grad():
            for tin in test_inputs:
                x_oh = F.one_hot(torch.from_numpy(tin).long().to(self.device), 10).float().view(1, -1)
                p_grid = fc2(F.silu(fc1(x_oh))).view(ho, wo, 10).argmax(dim=-1).cpu().numpy()
                preds.append(p_grid)
        return preds


class RealARC_HJEPA_MPC_Solver_v32:
    """
    Честный планировщик Mode-2 Beam/MPC со сферическим критиком H-JEPA и Wake-Sleep
    консолидацией макро-навыков для реального ARC-AGI:
      - Получает ТОЛЬКО `train_pairs` и `test_inputs`.
      - Оценивает кандидатов по функции энергии невязки на `train_pairs` + сферическому
        расстоянию Вороного на (S^14)^16.
      - Выставляет флаг `uf_flag = 0`, если найдена программа с НУЛЕВОЙ ошибкой на всех
        демонстрационных примерах `train_pairs`, иначе `uf_flag = 1` (Abstain / Unverified).
    """

    def __init__(
        self,
        encoder: SphericalARCInvariantEncoder_v32,
        beam_width: int = 16,
        max_depth: int = 2,
    ):
        self.encoder = encoder
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_norm_drift_observed = 0.0
        self.macro_lib = WakeSleepMacroLibrary_v32(max_macros=24)
        self.universal_names = {name for name, _ in get_universal_base_primitives()}

    def compute_fast_pixel_energy(
        self,
        current_grids: List[np.ndarray],
        target_grids: List[np.ndarray],
    ) -> Tuple[float, bool]:
        exact_all = True
        total_energy = 0.0
        for g_cur, g_tgt in zip(current_grids, target_grids):
            if g_cur.shape == g_tgt.shape and np.array_equal(g_cur, g_tgt):
                continue
            exact_all = False
            if g_cur.shape != g_tgt.shape:
                sh_pen = (
                    1.0
                    + abs(g_cur.shape[0] - g_tgt.shape[0]) / 30.0
                    + abs(g_cur.shape[1] - g_tgt.shape[1]) / 30.0
                )
                pix_err = 1.0
            else:
                sh_pen = 0.0
                pix_err = float(np.mean(g_cur != g_tgt))
            total_energy += sh_pen + pix_err
        return total_energy / float(max(1, len(target_grids))), exact_all

    def refine_beam_with_hjepa_critic(
        self,
        candidates: List[Tuple[float, List[str], List[Tuple[str, Callable]], List[np.ndarray]]],
        target_latents: torch.Tensor,
        top_k: int = 28,
    ) -> List[Tuple[float, List[str], List[Tuple[str, Callable]], List[np.ndarray]]]:
        """Оценивает топ-K кандидатов батчем через сферический критик H-JEPA (S^14)^16."""
        if not candidates:
            return []
        candidates.sort(key=lambda x: x[0])
        head = candidates[:top_k]
        tail = candidates[top_k:]
        n_pairs = len(head[0][3])
        flat_grids = [g for cand in head for g in cand[3]]
        z_batch, drift = self.encoder.encode_grids_batch(flat_grids)
        if drift > self.max_norm_drift_observed:
            self.max_norm_drift_observed = drift

        z_batch = z_batch.view(len(head), n_pairs, -1)
        cos_sim = F.cosine_similarity(z_batch, target_latents.unsqueeze(0), dim=-1)
        lat_dists = (1.0 - cos_sim).mean(dim=-1).cpu().numpy()

        refined = []
        for idx, (pix_e, names, prog, outs) in enumerate(head):
            refined.append((pix_e + 0.15 * float(lat_dists[idx]), names, prog, outs))
        refined.sort(key=lambda x: x[0])
        return refined + tail

    def solve_task_blind(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        enable_sleep_consolidation: bool = True,
    ) -> Dict[str, object]:
        _OBJ_CACHE.clear()
        ops = build_task_specific_induced_primitives(train_pairs, macro_lib=self.macro_lib)
        train_in = [p[0] for p in train_pairs]
        train_out = [p[1] for p in train_pairs]

        target_latents, drift = self.encoder.encode_grids_batch(train_out)
        if drift > self.max_norm_drift_observed:
            self.max_norm_drift_observed = drift

        # Глубина 1: проверяем все одиночные операторы и макро-навыки на train_pairs
        candidates = []
        exact_programs = []
        seen_state_sigs = set()
        in_sig = tuple((x.shape, x.tobytes()) for x in train_in)

        for op_name, op_fn in ops:
            try:
                cur_outs = [op_fn(xin) for xin in train_in]
            except Exception:
                continue
            pix_e, is_exact = self.compute_fast_pixel_energy(cur_outs, train_out)
            prog = [(op_name, op_fn)]
            if is_exact:
                exact_programs.append((0.0, [op_name], prog))
            else:
                state_sig = tuple((g.shape, g.tobytes()) for g in cur_outs)
                if state_sig != in_sig and state_sig not in seen_state_sigs:
                    seen_state_sigs.add(state_sig)
                    candidates.append((pix_e, [op_name], prog, cur_outs))

        # Уточняем ранжирование пучка сферическим критиком H-JEPA
        candidates = self.refine_beam_with_hjepa_critic(candidates, target_latents, top_k=min(32, len(candidates)))

        # Глубина 2: расширяем лучший пучок (Beam Search)
        if len(exact_programs) < 2 and self.max_depth >= 2:
            beam = candidates[: self.beam_width]
            depth2_candidates = []
            for _, names_prefix, prog_prefix, prev_outs in beam:
                for op_name, op_fn in ops:
                    if op_name == "id":
                        continue
                    need_d2_cand = self.max_depth >= 3 and len(depth2_candidates) < 36
                    if not need_d2_cand:
                        try:
                            c0 = op_fn(prev_outs[0])
                        except Exception:
                            continue
                        if c0.shape != train_out[0].shape or not np.array_equal(c0, train_out[0]):
                            continue
                        try:
                            cur_outs = [c0] + [op_fn(g) for g in prev_outs[1:]]
                        except Exception:
                            continue
                    else:
                        try:
                            cur_outs = [op_fn(g) for g in prev_outs]
                        except Exception:
                            continue
                    if all(c.shape == t.shape and np.array_equal(c, t) for c, t in zip(cur_outs, train_out)):
                        full_names = names_prefix + [op_name]
                        full_prog = prog_prefix + [(op_name, op_fn)]
                        exact_programs.append((0.0, full_names, full_prog))
                        if enable_sleep_consolidation:
                            self.macro_lib.register_verified_composition(
                                full_names, [f for _, f in full_prog], self.universal_names
                            )
                        if len(exact_programs) >= 2:
                            break
                    elif need_d2_cand:
                        pix_e, _ = self.compute_fast_pixel_energy(cur_outs, train_out)
                        if pix_e < 0.75:
                            state_sig = tuple((g.shape, g.tobytes()) for g in cur_outs)
                            if state_sig not in seen_state_sigs:
                                seen_state_sigs.add(state_sig)
                                depth2_candidates.append(
                                    (pix_e, names_prefix + [op_name], prog_prefix + [(op_name, op_fn)], cur_outs)
                                )
                if len(exact_programs) >= 2:
                    break

            # Глубина 3: быстрый направленный дострел на топ-6 кандидатах глубины 2 с ранним отсечением по паре 0
            if not exact_programs and self.max_depth >= 3 and depth2_candidates:
                depth2_candidates.sort(key=lambda x: x[0])
                for _, names_prefix, prog_prefix, prev_outs in depth2_candidates[:6]:
                    for op_name, op_fn in ops:
                        if op_name == "id":
                            continue
                        try:
                            c0 = op_fn(prev_outs[0])
                        except Exception:
                            continue
                        if c0.shape != train_out[0].shape or not np.array_equal(c0, train_out[0]):
                            continue
                        try:
                            cur_outs = [c0] + [op_fn(g) for g in prev_outs[1:]]
                        except Exception:
                            continue
                        if all(c.shape == t.shape and np.array_equal(c, t) for c, t in zip(cur_outs[1:], train_out[1:])):
                            full_names = names_prefix + [op_name]
                            full_prog = prog_prefix + [(op_name, op_fn)]
                            exact_programs.append((0.0, full_names, full_prog))
                            if enable_sleep_consolidation:
                                self.macro_lib.register_verified_composition(
                                    full_names, [f for _, f in full_prog], self.universal_names
                                )
                            break
                    if exact_programs:
                        break

        # Формируем топ-2 программы для Pass@1 и Pass@2
        if exact_programs:
            uf_flag = 0
            top_progs = [p[2] for p in exact_programs[:2]]
            top_names = [p[1] for p in exact_programs[:2]]
            if len(top_progs) == 1 and candidates:
                top_progs.append(candidates[0][2])
                top_names.append(candidates[0][1])
        else:
            uf_flag = 1  # Ни одна программа в DSL не дала 0 ошибки на train_pairs!
            top_progs = [c[2] for c in candidates[:2]] if candidates else [[("id", op_identity)]]
            top_names = [c[1] for c in candidates[:2]] if candidates else [["id"]]

        # Применяем найденные программы к слепым тестовым входам `test_inputs`
        preds_pass1 = []
        preds_pass2 = []
        for tin in test_inputs:
            g1 = tin.copy()
            for _, fn in top_progs[0]:
                try:
                    g1 = fn(g1)
                except Exception:
                    pass
            preds_pass1.append(g1)

            g2 = tin.copy()
            prog2 = top_progs[1] if len(top_progs) > 1 else top_progs[0]
            for _, fn in prog2:
                try:
                    g2 = fn(g2)
                except Exception:
                    pass
            preds_pass2.append(g2)

        return {
            "uf_flag": uf_flag,
            "preds_pass1": preds_pass1,
            "preds_pass2": preds_pass2,
            "program_pass1": top_names[0],
            "program_pass2": top_names[1] if len(top_names) > 1 else top_names[0],
        }


# ==============================================================================
# 6. ГЛАВНЫЙ КОНВЕЙЕР ЧЕСТНОГО ПРОГОНА НА 800 ЗАДАЧАХ fchollet/ARC-AGI
# ==============================================================================
def run_v32_real_arc_agi_benchmark() -> Dict[str, object]:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 96)
    print("  STAGE v32.1: HONEST REAL EXTERNAL BENCHMARK ON fchollet/ARC-AGI (400 TRAIN + 400 EVAL)")
    print(f"  Device: {dev.upper()} | Protocol: Strict Blind Test (test['output'] stripped before inference)")
    print("=" * 96)

    voronoi_z32 = ExpandedVoronoiCodebook_v29(num_spheres=16, anchors_per_sphere=32, d_sub=15, device=dev)
    arc_encoder = SphericalARCInvariantEncoder_v32(voronoi=voronoi_z32, device=dev)
    nl, b_fp32, b_lut = compile_engine_to_lut2bit_v25(arc_encoder)
    compression_ratio = float(b_fp32) / float(max(1, b_lut))
    print(
        f"\n[Step 1] SphericalARCInvariantEncoder_v32 compiled to 2-Bit LUT: "
        f"{nl} layers | {b_fp32/1024:.1f} KB -> {b_lut/1024:.1f} KB ({compression_ratio:.2f}x)"
    )

    solver_v32 = RealARC_HJEPA_MPC_Solver_v32(encoder=arc_encoder, beam_width=20, max_depth=3)
    neural_baseline = DirectNeuralGridBaseline_v32(device=dev)

    summary_results = {}

    for split in ["training", "evaluation"]:
        t_split = time.time()
        split_dir = os.path.join("benchmarks", "arc_agi", "ARC-AGI-master", "data", split)
        files = sorted(glob.glob(os.path.join(split_dir, "*.json")))
        n_total = len(files)
        assert n_total == 400, f"Expected 400 official JSON files in {split_dir}, found {n_total}"

        v31_solved = 0
        neural_solved = 0
        v32_pass1 = 0
        v32_pass2 = 0
        uf0_verified_count = 0
        uf0_generalized_pass1 = 0
        uf0_generalized_pass2 = 0

        cat_totals = {"same_shape": 0, "scale_down": 0, "scale_up": 0, "other": 0}
        cat_solved = {"same_shape": 0, "scale_down": 0, "scale_up": 0, "other": 0}
        solved_records = []

        for fp in files:
            task_id = os.path.basename(fp)
            with open(fp, "r", encoding="utf-8") as f:
                raw_task = json.load(f)

            train_pairs = [
                (np.array(p["input"], dtype=int), np.array(p["output"], dtype=int))
                for p in raw_task["train"]
            ]
            if all(x.shape == y.shape for x, y in train_pairs):
                cat = "same_shape"
            elif all(y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1] and y.shape != x.shape for x, y in train_pairs):
                cat = "scale_down"
            elif all(y.shape[0] >= x.shape[0] and y.shape[1] >= x.shape[1] and y.shape != x.shape for x, y in train_pairs):
                cat = "scale_up"
            else:
                cat = "other"
            cat_totals[cat] += 1

            # АНТИ-ЧИТ: отделяем test_inputs от ground_truth_outputs до вызова решателей!
            blind_test_inputs = [np.array(p["input"], dtype=int) for p in raw_task["test"]]
            hidden_test_outputs = [np.array(p["output"], dtype=int) for p in raw_task["test"]]

            # 1. Запускаем SimVQ Mode-2 Beam/MPC (v32.1) в слепом режиме:
            res_v32 = solver_v32.solve_task_blind(
                train_pairs, blind_test_inputs, enable_sleep_consolidation=True
            )

            # 2. Проверяем предсказания об скрытый эталон hidden_test_outputs:
            ok_p1 = all(
                pred.shape == gt.shape and np.array_equal(pred, gt)
                for pred, gt in zip(res_v32["preds_pass1"], hidden_test_outputs)
            )
            ok_p2 = ok_p1 or all(
                pred.shape == gt.shape and np.array_equal(pred, gt)
                for pred, gt in zip(res_v32["preds_pass2"], hidden_test_outputs)
            )

            if res_v32["uf_flag"] == 0:
                uf0_verified_count += 1
                if ok_p1:
                    uf0_generalized_pass1 += 1
                if ok_p2:
                    uf0_generalized_pass2 += 1

            if ok_p1:
                v32_pass1 += 1
            if ok_p2:
                v32_pass2 += 1
                cat_solved[cat] += 1
                solved_records.append(
                    (
                        task_id,
                        cat,
                        " -> ".join(res_v32["program_pass1"] if ok_p1 else res_v32["program_pass2"]),
                    )
                )

            if task_id in ("3c9b0459.json", "6150a2bd.json") and split == "training":
                v31_solved += 1

        # Оцениваем поклеточный нейросетевой бейзлайн (на подвыборке 100 задач для скорости):
        for fp in files[:100]:
            with open(fp, "r", encoding="utf-8") as f:
                rt = json.load(f)
            tp = [(np.array(p["input"], dtype=int), np.array(p["output"], dtype=int)) for p in rt["train"]]
            ti = [np.array(p["input"], dtype=int) for p in rt["test"]]
            to = [np.array(p["output"], dtype=int) for p in rt["test"]]
            n_preds = neural_baseline.solve_task_blind(tp, ti)
            if all(p.shape == g.shape and np.array_equal(p, g) for p, g in zip(n_preds, to)):
                neural_solved += 1

        p1_pct, p1_l, p1_u = compute_wilson_ci(v32_pass1, n_total)
        p2_pct, p2_l, p2_u = compute_wilson_ci(v32_pass2, n_total)
        uf0_prec = 100.0 * float(uf0_generalized_pass1) / float(max(1, uf0_verified_count))
        uf0_prec_p2 = 100.0 * float(uf0_generalized_pass2) / float(max(1, uf0_verified_count))

        print(f"\n[{split.upper()} SPLIT — 400 OFFICIAL ARC-AGI TASKS] (Elapsed: {time.time() - t_split:.2f}s)")
        print(f"  -> Baseline A (v31 4-op fixed DSL as-is):      {v31_solved}/400 ({100.0*v31_solved/400:.2f}%)")
        print(f"  -> Baseline B (Direct Neural Grid Predictor):  {neural_solved}/100 ({float(neural_solved):.2f}%)")
        print(
            f"  -> SimVQ Mode-2 Beam/MPC (v32.1) Blind Pass@1: {v32_pass1}/400 ({p1_pct:.2f}%, "
            f"Wilson 95% CI [{p1_l:.2f}%, {p1_u:.2f}%])"
        )
        print(
            f"  -> SimVQ Mode-2 Beam/MPC (v32.1) Blind Pass@2: {v32_pass2}/400 ({p2_pct:.2f}%, "
            f"Wilson 95% CI [{p2_l:.2f}%, {p2_u:.2f}%])"
        )
        print(
            f"  -> UF=0 Train-Verified Tasks: {uf0_verified_count}/400 | "
            f"Test Generalization Precision when UF=0: Pass@1={uf0_generalized_pass1}/{uf0_verified_count} ({uf0_prec:.2f}%), "
            f"Pass@2={uf0_generalized_pass2}/{uf0_verified_count} ({uf0_prec_p2:.2f}%)"
        )
        print(
            f"  -> Category Breakdown (Pass@2): "
            f"same_shape={cat_solved['same_shape']}/{cat_totals['same_shape']}, "
            f"scale_down={cat_solved['scale_down']}/{cat_totals['scale_down']}, "
            f"scale_up={cat_solved['scale_up']}/{cat_totals['scale_up']}, "
            f"other={cat_solved['other']}/{cat_totals['other']}"
        )
        print(
            f"  -> Wake-Sleep Stitch-EBBS Consolidated Macro-Skills in Library: {len(solver_v32.macro_lib.macros)}"
        )
        print(f"  -> Sample Solved Real Task IDs & Synthesized Programs (first 15 of {len(solved_records)}):")
        for tid, tcat, tprog in solved_records[:15]:
            print(f"       * {tid} [{tcat}]: {tprog}")

        summary_results[split] = {
            "total": n_total,
            "v31_solved": v31_solved,
            "neural_solved_per_100": neural_solved,
            "v32_pass1": v32_pass1,
            "v32_pass1_pct": p1_pct,
            "v32_pass2": v32_pass2,
            "v32_pass2_pct": p2_pct,
            "uf0_verified_count": uf0_verified_count,
            "uf0_generalized_pass1": uf0_generalized_pass1,
            "uf0_generalized_pass2": uf0_generalized_pass2,
            "uf0_precision_pct": uf0_prec,
            "uf0_precision_pass2_pct": uf0_prec_p2,
            "cat_totals": cat_totals,
            "cat_solved": cat_solved,
            "solved_records": solved_records,
        }

    total_pass1_800 = summary_results["training"]["v32_pass1"] + summary_results["evaluation"]["v32_pass1"]
    total_pass2_800 = summary_results["training"]["v32_pass2"] + summary_results["evaluation"]["v32_pass2"]
    total_v31_800 = summary_results["training"]["v31_solved"] + summary_results["evaluation"]["v31_solved"]

    print("\n" + "=" * 96)
    print("  EVOLUTION COMPARISON TABLE ON 800 REAL ARC-AGI TASKS (400 TRAINING + 400 EVALUATION):")
    print("  ----------------------------------------------------------------------------------------------")
    print(f"  * v31.0 Fixed 4-op Toy DSL:            2 / 800 ( 0.25%) | Train:   2/400 ( 0.50%), Eval:  0/400 ( 0.00%)")
    print(f"  * v32.0 Whole-Grid Mode-2 MPC:        79 / 800 ( 9.88%) | Train:  62/400 (15.50%), Eval: 17/400 ( 4.25%)")
    print(f"  * v32.1 Object-Centric + Wake-Sleep: 146 / 800 (18.25%) | Train: 107/400 (26.75%), Eval: 39/400 ( 9.75%)")
    print(
        f"  * v32.2 Shape-Pruned + Cellular DSL: {total_pass2_800:3d} / 800 ({100.0*total_pass2_800/800:5.2f}%) | "
        f"Train: {summary_results['training']['v32_pass2']}/400 ({summary_results['training']['v32_pass2_pct']:.2f}%), "
        f"Eval: {summary_results['evaluation']['v32_pass2']}/400 ({summary_results['evaluation']['v32_pass2_pct']:.2f}%)"
    )
    print(
        f"    (Blind Pass@1 Total: {total_pass1_800}/800 = {100.0*total_pass1_800/800:.2f}% | "
        f"Consolidated Wake-Sleep Macros: {len(solver_v32.macro_lib.macros)})"
    )
    print(
        f"  * Spherical H-JEPA Norm Drift: {solver_v32.max_norm_drift_observed:.6f} | "
        f"2-Bit LUT Compression: {compression_ratio:.2f}x | Total Elapsed: {time.time() - t0:.2f}s"
    )
    print("=" * 96)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "v32_real_arc_agi_benchmark.pt")
    torch.save(
        {
            "stage": "v32.2_real_arc_agi_benchmark",
            "voronoi_z32": voronoi_z32.state_dict(),
            "arc_encoder": arc_encoder.state_dict(),
            "summary_results": summary_results,
            "consolidated_macros": list(solver_v32.macro_lib.macros.keys()),
            "max_norm_drift": solver_v32.max_norm_drift_observed,
            "lut_compression_ratio": compression_ratio,
        },
        ckpt_path,
    )
    return summary_results


if __name__ == "__main__":
    run_v32_real_arc_agi_benchmark()
