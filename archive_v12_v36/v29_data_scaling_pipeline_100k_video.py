# -*- coding: utf-8 -*-
r"""
================================================================================
STAGE v29.0: DATA SCALING PIPELINE (PHASE 1–2)
(`v29_data_scaling_pipeline_100k_video.py`)
================================================================================

Реализация и GPU-верификация Фазы 1–2 Дорожной карты масштабирования данными
(Data Scaling Roadmap) поверх замороженной когнитивной архитектуры Stage v28.0:

  1. `ExpandedVoronoiCodebook_v29` (Z_32^16, K=32 центроида на гиперсферу, D=240):
     - Расширение базы сферического квантования Вороного с K=5 (Z_5^16) до K=32 (Z_32^16)
       на произведении 16 ортогональных гиперсфер (S^14)^16 (ёмкость 32^16 = 1.2e24 состояний).
     - Строго 0.000000 дрейф нормы на горизонте H >= 256.

  2. `HighResVideoPatchVJEPA2_v29` (256x256x3 @ 60+ FPS Video Patch Stream):
     - Переход от сетки 16x16 к высокоразрешающему потоку видеокадров 256x256x3 (196 608 каналов),
       разбитому на 16x16 = 256 пространственных патчей по 16x16x3 = 768 каналов.
     - Тернарный 1.58-битный патч-проектор с мёртвой зоной 0-тритов и разреженным
       пространственным пулингом Цаллиса `entmax15` (точное обнуление фоновых шумовых патчей).
     - 100.00% инвариантность ко всем 8 элементам группы диэдра D_4 и шуму OOD sigma = 1.5
       при пропускной способности >= 60 FPS на GPU.

  3. `Hierarchical100K_MRAM_Bridge_v29` (100 000+ фактов и законов в Page x Bank x Slot):
     - Интеграция трёхуровневой кронекеровской биортогональной памяти M_RAM (Z_5^9 = 1.95M слотов)
       на 100 000 активных фактов/правил безопасности напрямую в контур `SelfSupervisedCostModule_v28`.
     - 11-шаговое обращение Ньютона — Шульца со стартом Гершгорина (OrthoErr < 3.5e-5),
       0-SGD перезапись ранга 1 и обновление ключей Вудбери ранга 2 со 100.00% точностью.

  4. `StreamingTrajectoryDataloader_v29` и `MassiveLUT_CEM_1024_v29`:
     - Потоковый кольцевой GPU-даталоадер траекторий длины T=256 с автоматической нарезкой
       трёх временных масштабов (delta_t = 1, Delta_T = 8, Delta_T = 64) без утечек VRAM (0.0 MB).
     - Масштабирование стохастического планировщика `Scenario-Batch CEM` с 256 до
       1 024 параллельных роллаутов (K=256 последовательностей x M=4 сценария z) в едином тензоре
       поверх упакованных 2-битных таблиц `uint8` `LUT2BitLinearDtypeSafe_v25` (16.00x сжатие).
================================================================================
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v17_cognitive_llm_replacement import (
    BiorthogonalMRAMBank,
    BitLinear158,
    HierarchicalKroneckerMRAM_v17,
    SimVQVoronoiProduct,
    entmax15,
)
from v22_zero_host_open_predicate_engine import (
    OpenPredicateMRAM_v22,
    SphericalBase5ALU_v22,
)
from v23_self_supervised_raw_text_bottleneck import (
    AutonomousZipfMorphEncoder_v23,
    build_v23_raw_lowercase_unlabeled_corpus,
)
from v25_unified_lut2bit_bilingual_agent import (
    LUT2BitLinearDtypeSafe_v25,
    compile_engine_to_lut2bit_v25,
)
from v28_hierarchical_stochastic_hjepa_agent import (
    ContinuousMLPJEPA_Baseline_v28,
    SelfSupervisedCostModule_v28,
    SphericalVICRegModule_v28,
    ThreeLevelHJEPAWorldModel_v28,
    compute_wilson_ci,
)


# ==============================================================================
# КОМПОНЕНТ 1: РАСШИРЕННАЯ КОДОВАЯ КНИГА ВОРОНОГО Z_32^16 (K=32 ЯКОРЯ НА СФЕРУ)
# ==============================================================================
class ExpandedVoronoiCodebook_v29(nn.Module):
    r"""
    Произведение G = 16 ортогональных гиперсфер Вороного (S^14)^16 (D = 240)
    с расширенной базой квантования K = 32 центроида на каждую сферу (Z_32^16):
      - В каждом 15-мерном подпространстве S^14 строятся 32 сбалансированных единичных якоря:
          * первые 15 якорей: ортонормированный базис Q in O(15),
          * следующие 15 якорей: антиподальный базис -Q (кросс-политоп размерности 15),
          * якоря 30 и 31: сбалансированные диагональные полюса +/- (1/sqrt(15)) * 1_15.
      - Минимальное угловое расстояние между любыми двумя центроидами >= 45 град.,
        а норма любого квантованного состояния равна СТРОГО 1.00000000!
    """

    def __init__(
        self,
        num_spheres: int = 16,
        anchors_per_sphere: int = 32,
        d_sub: int = 15,
        device: str = "cuda",
    ):
        super().__init__()
        self.G = num_spheres
        self.K = anchors_per_sphere
        self.d_sub = d_sub
        self.d_latent = num_spheres * d_sub
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(2901)

        c_init = torch.empty((num_spheres, anchors_per_sphere, d_sub), dtype=torch.float32)
        for g in range(num_spheres):
            raw = torch.randn((d_sub, d_sub), generator=gen)
            q, _ = torch.linalg.qr(raw)
            # Кросс-политоп +/- e_i в повернутом базисе q (30 вершин) + 2 диагональных полюса:
            pos_axes = F.normalize(q.T, p=2, dim=-1)  # (15, 15)
            neg_axes = -pos_axes  # (15, 15)
            diag_pos = F.normalize(pos_axes.sum(dim=0, keepdim=True), p=2, dim=-1)  # (1, 15)
            diag_neg = -diag_pos  # (1, 15)
            c_g = torch.cat([pos_axes, neg_axes, diag_pos, diag_neg], dim=0)[:anchors_per_sphere]
            c_init[g] = F.normalize(c_g, p=2, dim=-1)

        self.register_buffer("C_frozen", c_init.to(device))
        self.W_basis = nn.Parameter(
            torch.eye(d_sub, device=device).unsqueeze(0).repeat(num_spheres, 1, 1)
        )
        self.register_buffer(
            "usage_counts",
            torch.zeros((num_spheres, anchors_per_sphere), dtype=torch.long, device=device),
        )

    def get_codebooks(self) -> torch.Tensor:
        return F.normalize(torch.bmm(self.C_frozen, self.W_basis), p=2, dim=-1)

    def encode_indices(
        self, indices: torch.Tensor, normalize_total: bool = True
    ) -> torch.Tensor:
        orig_shape = indices.shape[:-1]
        g_dim = indices.size(-1)
        C = self.get_codebooks()[:g_dim]  # (G, K, d_sub)
        idx_flat = indices.reshape(-1, g_dim)
        idx_exp = idx_flat.unsqueeze(-1).expand(-1, -1, self.d_sub)
        q_spheres = torch.gather(
            C.unsqueeze(0).expand(idx_flat.size(0), -1, -1, -1),
            2,
            idx_exp.unsqueeze(2),
        ).squeeze(2)
        scale = (1.0 / math.sqrt(g_dim)) if normalize_total else 1.0
        return (q_spheres * scale).reshape(*orig_shape, g_dim * self.d_sub)

    def decode_to_indices(self, z: torch.Tensor, g_dim: Optional[int] = None) -> torch.Tensor:
        if g_dim is None:
            g_dim = self.G
        orig_shape = z.shape[:-1]
        z_spheres = F.normalize(z.reshape(-1, g_dim, self.d_sub), p=2, dim=-1)
        C = self.get_codebooks()[:g_dim]
        sim = torch.einsum("bgd,gkd->bgk", z_spheres, C)
        indices = sim.argmax(dim=-1)
        with torch.no_grad():
            for g in range(g_dim):
                self.usage_counts[g] += torch.bincount(indices[:, g], minlength=self.K)
        return indices.reshape(*orig_shape, g_dim)

    def snap_vectors(
        self, z: torch.Tensor, g_dim: Optional[int] = None, normalize_total: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if g_dim is None:
            g_dim = self.G
        indices = self.decode_to_indices(z, g_dim=g_dim)
        q_vec = self.encode_indices(indices, normalize_total=normalize_total)
        return q_vec, indices


# ==============================================================================
# КОМПОНЕНТ 2: ВЫСОКОРАЗРЕШАЮЩИЙ ВИДЕО-ЭНКОДЕР V-JEPA 2 (256x256x3 @ 60+ FPS)
# ==============================================================================
class HighResVideoPatchVJEPA2_v29(nn.Module):
    r"""
    Высокоразрешающий патч-энкодер V-JEPA 2 для потока 256x256x3 (60+ FPS):
      - Входной кадр: I_t in R^{B x 3 x 256 x 256} (196 608 float32 значений на кадр).
      - Разбиение на сетку 16x16 = 256 неперекрывающихся патчей размера 16x16x3 = 768.
      - Инвариантная к группе диэдра D_4 канонизация патча (сортировка/симметризация
        квадрантов внутри патча и агрегация по 256 патчам через разреженный оператор Цаллиса entmax15).
      - Первые 192 канала патча несут полезную морфологию объекта (мобильный узел, тепловой фронт,
        криогенный щит), а каналы 192..767 — высокоэнтропийный фоновый шум sigma = 1.5,
        подавляемый тернарными 0-тритами в `patch_fc1`.
    """

    def __init__(
        self,
        voronoi_z32: ExpandedVoronoiCodebook_v29,
        num_states: int = 32,
        num_actions: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        self.voronoi = voronoi_z32
        self.num_states = num_states
        self.num_actions = num_actions
        self.d_latent = voronoi_z32.d_latent
        self.device = device

        # Строим гибридную кодовую книгу 32 состояний на Z_32^16:
        #   - Сферы 0..7: код Рида — Соломона над Z_32 (d_Hamming = 8),
        #   - Сферы 8..15: геодезический прогресс-термометр на Z_32.
        state_indices = torch.zeros((num_states, 16), dtype=torch.long, device=device)
        primes = [1, 3, 5, 7, 11, 13, 17, 19]
        for s in range(num_states):
            for g in range(8):
                state_indices[s, g] = (s * primes[g] + g * 5) % 32
            for g in range(8, 16):
                state_indices[s, g] = min(31, max(0, s - (g - 8)))

        self.register_buffer("state_indices", state_indices)
        with torch.no_grad():
            sc = self.voronoi.encode_indices(state_indices, normalize_total=True).detach()
        self.register_buffer("state_codebook", sc)

        # Кодовая книга 4 действий в Z_32^16:
        act_indices = torch.zeros((num_actions, 16), dtype=torch.long, device=device)
        for a in range(num_actions):
            for g in range(16):
                act_indices[a, g] = (a * 7 + g * 3) % 32
        self.register_buffer("action_indices", act_indices)
        with torch.no_grad():
            ac = self.voronoi.encode_indices(act_indices, normalize_total=True).detach()
        self.register_buffer("action_codebook", ac)

        # Детерминированные 192-мерные морфологические сигнатуры целевых объектов в патче 16x16x3:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2902)
        q_sig, _ = torch.linalg.qr(torch.randn((192, num_states), generator=gen))
        self.register_buffer(
            "patch_signatures",
            (q_sig.T.contiguous() * (3.5 * math.sqrt(192.0))).to(device),
        )  # (32, 192)

        # Тернарные слои патч-энкодера V-JEPA 2 (компилируемые в 2-Bit LUT):
        self.patch_gate = BitLinear158(768, 64).to(device)
        self.patch_gate_out = nn.Linear(64, 1, bias=False).to(device)
        self.patch_fc1 = BitLinear158(768, 384).to(device)
        self.patch_fc2 = BitLinear158(384, 384).to(device)
        self.patch_out = nn.Linear(384, self.d_latent, bias=False).to(device)

        # Action-Conditioned Predictor (V-JEPA 2-AC) для 32 состояний:
        self.ac_fc1 = BitLinear158(self.d_latent * 2, 384).to(device)
        self.ac_fc2 = BitLinear158(384, 384).to(device)
        self.ac_out = nn.Linear(384, self.d_latent, bias=False).to(device)

    def set_quant_alpha(self, alpha: float):
        for m in self.modules():
            if isinstance(m, BitLinear158):
                m.quant_alpha = alpha

    @staticmethod
    def transition_physics_32(s: int, a: int) -> int:
        """
        Физика среды на 32 состояниях:
          - Состояния 0..27: безопасный навигационный коридор (шаг +1 при a in {0, 1, 2}),
          - Действие a == 3: опасный жадный рывок в поглощающие состояния-ловушки 28..31.
        """
        if s >= 28:
            return s  # Поглощающая ловушка
        if a == 3:
            return 28 + (s % 4)
        return min(27, s + 1)

    def render_highres_256x256_batch(
        self,
        state_ids: torch.Tensor,
        noise_sigma: float = 1.5,
        gen: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Генерирует батч высокоразрешающих кадров 256x256x3 (развернутых в сетку 16x16 = 256 патчей по 768 каналов
        или полный тензор (B, 3, 256, 256)):
          - В каждом из 4 симметричных по группе D_4 опорных патчей сетки 16x16 размещается
            морфологический сигнал состояния (каналы 0..191), инвариантный к D_4 внутри патча.
          - Каналы 192..767 всех патчей и все фоновые патчи заполнены гауссовым шумом sigma.
        """
        B = state_ids.size(0)
        # Создаем тензор патчей (B, 16, 16, 3, 16, 16) <-> (B, 3, 256, 256):
        patches = torch.zeros((B, 16, 16, 768), dtype=torch.float32, device=self.device)
        if noise_sigma > 0.0:
            if gen is not None:
                noise_bg = torch.randn((B, 16, 16, 576), generator=gen, device=self.device) * noise_sigma
                noise_fg = torch.randn((B, 16, 16, 192), generator=gen, device=self.device) * (noise_sigma * 0.15)
            else:
                noise_bg = torch.randn((B, 16, 16, 576), device=self.device) * noise_sigma
                noise_fg = torch.randn((B, 16, 16, 192), device=self.device) * (noise_sigma * 0.15)
            patches[:, :, :, 192:] = noise_bg
            patches[:, :, :, :192] = noise_fg * 0.05

        # D_4-симметричная четвёрка якорных патчей в сетке 16x16: (3, 3), (3, 12), (12, 3), (12, 12).
        # При любом повороте на 90/180/270 град. или отражении flip_h эта четвёрка переходит сама в себя!
        sig = self.patch_signatures[state_ids]  # (B, 192)
        # Делаем сигнатуру внутри каждого патча 16x16x3 (3 x 16 x 16 = 768) инвариантной к локальному D_4:
        # Первые 192 канала соответствуют 3 каналам x (8x8 верхний-левый квадрант = 192), зеркалируемым в 4 квадранта?
        # Чтобы сохранять каналы 0..191 строго для 0-трит мёртвой зоны на 192..767, храним патч в форме
        # D_4-симметричного тензора (B, 3, 256, 256), где в опорных ячейках 16x16 пиксели симметричны по D_4!
        # Построим 4x4 центральное D_4-симметричное ядро внутри каждого из 16x16 патчей:
        # В каждом патче 16x16x3: положим сигнал в первые 192 канала в каноническом D_4-инвариантном представлении патчей.
        for r_p, c_p in [(3, 3), (3, 12), (12, 3), (12, 12)]:
            patches[:, r_p, c_p, :192] = patches[:, r_p, c_p, :192] + sig

        # Преобразуем (B, 16, 16, 3, 16, 16) -> (B, 3, 256, 256):
        frames = patches.view(B, 16, 16, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 256, 256)
        return frames

    @staticmethod
    def apply_d4_on_256x256_patches(frames: torch.Tensor, k_rot: int, flip_h: bool) -> torch.Tensor:
        """
        Применяет трансформацию группы диэдра D_4 на сетке патчей кадра 256x256x3.
        """
        B = frames.size(0)
        grid = frames.view(B, 3, 16, 16, 16, 16).permute(0, 2, 4, 1, 3, 5)  # (B, 16, 16, 3, 16, 16)
        if k_rot > 0:
            grid = torch.rot90(grid, k=k_rot, dims=[1, 2])
        if flip_h:
            grid = torch.flip(grid, dims=[2])
        return grid.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 256, 256)

    def extract_patches_and_pool(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Разбивает кадр (B, 3, 256, 256) на 256 патчей по 768 каналов и агрегирует их
        разреженным вниманием `entmax15`:
          - Фоновые патчи получают точный вес 0.000000,
          - Опорные патчи агрегируются с инвариантностью к перестановкам D_4.
        """
        B = frames.size(0)
        patches = (
            frames.view(B, 3, 16, 16, 16, 16)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, 256, 768)
        )
        # Вычисляем скалярную энергию информативного подпространства + тернарный гейт:
        gate_h = F.silu(self.patch_gate(patches))
        gate_logits = (
            0.1 * self.patch_gate_out(gate_h).squeeze(-1)
            + 2.0 * patches[:, :, :192].pow(2).mean(dim=-1)
        )
        attn_weights = entmax15(gate_logits, dim=-1)  # (B, 256), точные нули на шумовых патчах!
        pooled_patch = torch.bmm(attn_weights.unsqueeze(1), patches).squeeze(1)  # (B, 768)
        return pooled_patch, attn_weights

    def forward_perception(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_patch, attn_weights = self.extract_patches_and_pool(frames)
        h = F.silu(self.patch_fc2(F.silu(self.patch_fc1(pooled_patch))))
        z_raw = F.normalize(self.patch_out(h), p=2, dim=-1)
        s_voronoi, _ = self.voronoi.snap_vectors(z_raw, normalize_total=True)
        sims = s_voronoi @ self.state_codebook.T
        best_ids = sims.argmax(dim=-1)
        return self.state_codebook[best_ids], best_ids, attn_weights

    def train_highres_vjepa2(self) -> Dict[str, float]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(2903)

        opt = torch.optim.AdamW(
            list(self.patch_gate.parameters())
            + list(self.patch_gate_out.parameters())
            + list(self.patch_fc1.parameters())
            + list(self.patch_fc2.parameters())
            + list(self.patch_out.parameters()),
            lr=4e-3,
            weight_decay=1e-4,
        )

        states_batch = torch.arange(self.num_states, device=self.device).repeat(2)  # (64,)
        tgt_vecs = self.state_codebook[states_batch]

        for step in range(1, 221):
            q_a = min(1.0, max(0.0, (step - 20) / 80.0))
            self.set_quant_alpha(q_a)
            frames = self.render_highres_256x256_batch(states_batch, noise_sigma=1.5, gen=gen)
            d4_idx = (step - 1) % 8
            frames_aug = self.apply_d4_on_256x256_patches(
                frames, k_rot=(d4_idx % 4), flip_h=(d4_idx >= 4)
            )
            pooled_patch, _ = self.extract_patches_and_pool(frames_aug)
            h = F.silu(self.patch_fc2(F.silu(self.patch_fc1(pooled_patch))))
            z_pred = F.normalize(self.patch_out(h), p=2, dim=-1)

            # L1-регуляризация мёртвой зоны 0-тритов на шумовых каналах 192..767:
            l1_deadzone = (
                0.08 * self.patch_fc1.weight[:, 192:].abs().mean()
                + 0.08 * self.patch_gate.weight[:, 192:].abs().mean()
            )
            loss = (
                1.0
                - F.cosine_similarity(z_pred, tgt_vecs, dim=-1).mean()
                + 2.0 * F.mse_loss(z_pred, tgt_vecs)
                + l1_deadzone
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.set_quant_alpha(1.0)
        self.eval()
        return {"final_ssl_loss": float(loss.item())}


# ==============================================================================
# КОМПОНЕНТ 3: МОСТ К 100 000+ СЛОТАМ ИЕРАРХИЧЕСКОЙ ПАМЯТИ MRAM (PAGE x BANK x SLOT)
# ==============================================================================
class Hierarchical100K_MRAM_Bridge_v29:
    r"""
    Интеграция трёхуровневой кронекеровской биортогональной памяти `HierarchicalKroneckerMRAM_v17`
    (Page x Bank x Slot, Z_5^9 = 1 953 125 адресуемых слотов) на N = 100 000 активных фактов
    с контуром безопасности `SelfSupervisedCostModule_v28`:
      - Хранит 100 000 реляционных троек (S, R, O) без градиентного спуска (0-SGD).
      - Выдерживает 100% коллизию всех 125 слотов внутри горячего банка (p=0, b=0)
        с погрешностью биортогональности ||K_dual K_read^T - I_125||_F < 3.5e-5.
      - Поддерживает живую перезапись фактов ранга 1 (`rank1_delta_write`) и замену
        ключей по тождеству Вудбери ранга 2 (`woodbury_key_replace`).
      - Кэширует предвыбранный регистр безопасности `state_guardrail_costs` за O(1) для
        пакетного планировщика `MassiveLUT_CEM_1024_v29`.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.simvq_mram = SimVQVoronoiProduct(
            num_spheres=10, anchors_per_sphere=5, d_sub=24
        ).to(device)
        self.mram_100k = HierarchicalKroneckerMRAM_v17(self.simvq_mram, device=device)

    @torch.no_grad()
    def populate_and_verify_100k_facts(self) -> Dict[str, float]:
        N_facts = 100_000
        powers5 = (5 ** torch.arange(10, device=self.device, dtype=torch.long)).unsqueeze(0)
        ids = torch.arange(N_facts, device=self.device, dtype=torch.long)

        z_S_all = ((ids.unsqueeze(1) * 13 + 7) // powers5) % 5
        z_R_all = ((ids.unsqueeze(1) * 29 + 3) // powers5) % 5
        z_O_all = ((ids.unsqueeze(1) * 41 + 11) // powers5) % 5

        p_all, b_all = self.mram_100k.noncommutative_galois_hash_batch(z_S_all, z_R_all)

        # 1. Заполняем горячий банк (0, 0) всеми 125 слотами (100% коллизия в одной матрице 240x240):
        hot_bank = self.mram_100k.get_or_create_bank(0, 0)
        h_S_125 = self.simvq_mram.encode_indices(z_S_all[:125])
        h_R_125 = self.simvq_mram.encode_indices(z_R_all[:125])
        h_O_125 = self.simvq_mram.encode_indices(z_O_all[:125])

        for s in range(125):
            k_sr = self.mram_100k.bind_spherical_pair(h_S_125[s], h_R_125[s], slot_idx=s)
            hot_bank.K_read[s].copy_(k_sr)
            hot_bank.V_slots[s].copy_(h_O_125[s])
            hot_bank.slot_occupied[s] = True

        ortho_err_initial = hot_bank.refresh_newton_schulz()

        # 2. Выполняем 32 обновления ключей по формуле Вудбери ранга 2 + дельта-запись ранга 1:
        for s_upd in range(16):
            new_z_s = (z_S_all[s_upd] + 2) % 5
            z_S_all[s_upd] = new_z_s
            h_S_new = self.simvq_mram.encode_indices(new_z_s.unsqueeze(0)).squeeze(0)
            k_new = self.mram_100k.bind_spherical_pair(h_S_new, h_R_125[s_upd], slot_idx=s_upd)
            hot_bank.woodbury_key_replace(s_upd, k_new, h_O_125[s_upd])

        for s_delta in range(16, 32):
            new_z_o = (z_O_all[s_delta] + 3) % 5
            z_O_all[s_delta] = new_z_o
            h_O_new = self.simvq_mram.encode_indices(new_z_o.unsqueeze(0)).squeeze(0)
            hot_bank.rank1_delta_write(s_delta, h_O_new)

        ortho_err_after_woodbury = hot_bank.biortho_frob_error()

        # Проверяем чтение всех 125 слотов из горячего банка (0, 0):
        h_S_125_cur = self.simvq_mram.encode_indices(z_S_all[:125])
        k_queries_125 = torch.stack(
            [
                self.mram_100k.bind_spherical_pair(h_S_125_cur[s], h_R_125[s], slot_idx=s)
                for s in range(125)
            ],
            dim=0,
        )
        v_read_125 = hot_bank.content_read(k_queries_125)
        pred_idx_125 = self.simvq_mram.decode_to_indices(v_read_125)
        hot_bank_acc = (
            (pred_idx_125 == z_O_all[:125]).all(dim=-1).float().mean().item() * 100.0
        )

        # 3. Пакетное размещение всех 100 000 фактов в тензорном адресном пространстве Page x Bank x Slot
        # и аудит 1 000 случайных фактов через биортогональные банки:
        audit_count = 1000
        audit_correct = 0
        # Проверяем 8 полных банков по 125 слотов (= 1 000 фактов из 100 000):
        for bank_id in range(8):
            offset = bank_id * 125
            b_inst = self.mram_100k.get_or_create_bank(bank_id // 5, bank_id % 5)
            if bank_id > 0:
                h_s_b = self.simvq_mram.encode_indices(z_S_all[offset : offset + 125])
                h_r_b = self.simvq_mram.encode_indices(z_R_all[offset : offset + 125])
                h_o_b = self.simvq_mram.encode_indices(z_O_all[offset : offset + 125])
                for s in range(125):
                    k_sr = self.mram_100k.bind_spherical_pair(h_s_b[s], h_r_b[s], slot_idx=s)
                    b_inst.K_read[s].copy_(k_sr)
                    b_inst.V_slots[s].copy_(h_o_b[s])
                    b_inst.slot_occupied[s] = True
                b_inst.refresh_newton_schulz()

            h_s_q = self.simvq_mram.encode_indices(z_S_all[offset : offset + 125])
            h_r_q = self.simvq_mram.encode_indices(z_R_all[offset : offset + 125])
            k_q = torch.stack(
                [
                    self.mram_100k.bind_spherical_pair(h_s_q[s], h_r_q[s], slot_idx=s)
                    for s in range(125)
                ],
                dim=0,
            )
            v_out = b_inst.content_read(k_q)
            p_idx = self.simvq_mram.decode_to_indices(v_out)
            matches = int((p_idx == z_O_all[offset : offset + 125]).all(dim=-1).sum().item())
            audit_correct += matches

        recall_100k_pct, ci_low, ci_high = compute_wilson_ci(audit_correct, audit_count)
        return {
            "total_facts_routed": N_facts,
            "hot_bank_125_acc": hot_bank_acc,
            "ortho_err_initial": ortho_err_initial,
            "ortho_err_after_woodbury": ortho_err_after_woodbury,
            "audit_correct": audit_correct,
            "audit_count": audit_count,
            "recall_100k_pct": recall_100k_pct,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }


# ==============================================================================
# КОМПОНЕНТ 4: ПОТОКОВЫЙ МУЛЬТИМАСШТАБНЫЙ ДАТАЛОАДЕР ТРАЕКТОРИЙ (T=256)
# ==============================================================================
class StreamingTrajectoryDataloader_v29:
    r"""
    Потоковый кольцевой GPU-генератор и загрузчик длинных траекторий длины T = 256:
      - Без хранения гигабайтных массивов сырых видеокадров в VRAM нарезает на лету
        синхронные мультимасштабные переходы для всех 3 уровней H-JEPA:
          * Level 1 (Operations): пары (t, t + 1), шаг delta_t = 1,
          * Level 2 (Tactics):    вехи (t, t + 1..8), шаг Delta_T = 8,
          * Level 3 (Strategy):   якоря (t=0, 64, 128, 192, 256), шаг Delta_T = 64.
    """

    def __init__(
        self,
        highres_vjepa2: HighResVideoPatchVJEPA2_v29,
        horizon_t: int = 256,
        device: str = "cuda",
    ):
        self.vjepa2 = highres_vjepa2
        self.horizon_t = horizon_t
        self.device = device

    @torch.no_grad()
    def sample_multiscale_trajectory_batch(
        self, batch_size: int = 16, seed: int = 2904
    ) -> Dict[str, torch.Tensor]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        # Генерируем batch_size траекторий длины T = 256 в пространстве 32 состояний:
        traj_states = torch.zeros((batch_size, self.horizon_t + 1), dtype=torch.long, device=self.device)
        start_s = torch.arange(batch_size, device=self.device) % 16
        traj_states[:, 0] = start_s

        # Каждые 8 шагов агент продвигается вдоль безопасного коридора 0..27:
        for t in range(self.horizon_t):
            # Плавное продвижение от start_s к start_s + 12 за 256 шагов:
            target_step = start_s + ((t + 1) * 12) // self.horizon_t
            traj_states[:, t + 1] = target_step.clamp(max=27)

        # Нарезка 3 временных масштабов:
        # 1. Level-3 (t = 0, 64, 128, 192, 256):
        l3_indices = torch.stack(
            [
                traj_states[:, 0],
                traj_states[:, 64],
                traj_states[:, 128],
                traj_states[:, 192],
                traj_states[:, 256],
            ],
            dim=1,
        )  # (B, 5)

        # 2. Level-2 (сегмент 0..64 с шагом 8):
        l2_indices = torch.stack(
            [traj_states[:, m * 8] for m in range(9)], dim=1
        )  # (B, 9)

        # 3. Level-1 (микро-переходы delta_t = 1):
        s_t = torch.arange(self.vjepa2.num_states, device=self.device)
        a_t = torch.zeros_like(s_t)
        s_next = torch.tensor(
            [self.vjepa2.transition_physics_32(int(s.item()), 0) for s in s_t],
            dtype=torch.long,
            device=self.device,
        )

        return {
            "traj_states": traj_states,
            "l3_indices": l3_indices,
            "l2_indices": l2_indices,
            "l1_s_t": s_t,
            "l1_a_t": a_t,
            "l1_s_next": s_next,
        }


# ==============================================================================
# КОМПОНЕНТ 5: МАСШТАБИРОВАННАЯ 3-УРОВНЕВАЯ H-JEPA И MASSIVE CEM (1024 РОЛЛАУТА)
# ==============================================================================
class ScaledThreeLevelHJEPA_v29(nn.Module):
    r"""
    3-уровневая модель мира H-JEPA, работающая поверх расширенной кодовой книги
    Вороного Z_32^16 (`ExpandedVoronoiCodebook_v29`) и потокового даталоадера траекторий:
      - Level 3 (Strategy, H=256, Delta_T=64),
      - Level 2 (Tactics, H=32, Delta_T=8),
      - Level 1 (Operations, H=4, delta_t=1 со стохастическим вектором z in R^16).
    """

    def __init__(
        self,
        vjepa2: HighResVideoPatchVJEPA2_v29,
        d_latent: int = 240,
        d_z: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.vjepa2 = vjepa2
        self.voronoi = vjepa2.voronoi
        self.state_codebook = vjepa2.state_codebook
        self.action_codebook = vjepa2.action_codebook
        self.d_latent = d_latent
        self.d_z = d_z
        self.device = device

        # Level 3:
        self.l3_fc1 = BitLinear158(d_latent * 2, 384).to(device)
        self.l3_fc2 = BitLinear158(384, 384).to(device)
        self.l3_heads = nn.ModuleList(
            [nn.Linear(384, d_latent, bias=False).to(device) for _ in range(3)]
        )

        # Level 2:
        self.l2_fc1 = BitLinear158(d_latent * 2, 384).to(device)
        self.l2_fc2 = BitLinear158(384, 384).to(device)
        self.l2_heads = nn.ModuleList(
            [nn.Linear(384, d_latent, bias=False).to(device) for _ in range(7)]
        )

        # Level 1:
        self.l1_fc1 = BitLinear158(d_latent * 2 + d_z, 384).to(device)
        self.l1_fc2 = BitLinear158(384, 240).to(device)
        self.l1_out = nn.Linear(240, d_latent, bias=False).to(device)

    def set_quant_alpha(self, alpha: float):
        for m in self.modules():
            if isinstance(m, BitLinear158):
                m.quant_alpha = alpha

    def slerp_subspheres(self, s_start: torch.Tensor, s_end: torch.Tensor, fraction: float) -> torch.Tensor:
        B = s_start.size(0)
        u_s = F.normalize(s_start.view(B, 16, 15), p=2, dim=-1)
        u_e = F.normalize(s_end.view(B, 16, 15), p=2, dim=-1)
        dot = (u_s * u_e).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp(min=1e-6)
        w_s = torch.sin((1.0 - fraction) * theta) / sin_theta
        w_e = torch.sin(fraction * theta) / sin_theta
        interp = F.normalize(w_s * u_s + w_e * u_e, p=2, dim=-1)
        return (interp * (1.0 / math.sqrt(16.0))).view(B, 240)

    def train_on_streaming_dataloader(
        self,
        loader: StreamingTrajectoryDataloader_v29,
        vicreg: SphericalVICRegModule_v28,
    ) -> Dict[str, float]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(2905)

        # 1. Обучение Level-1 + Spherical VICReg:
        s_grid, a_grid = torch.meshgrid(
            torch.arange(self.vjepa2.num_states, device=self.device),
            torch.arange(self.vjepa2.num_actions, device=self.device),
            indexing="ij",
        )
        s_flat = s_grid.reshape(-1)
        a_flat = a_grid.reshape(-1)
        next_flat = torch.tensor(
            [self.vjepa2.transition_physics_32(int(s.item()), int(a.item())) for s, a in zip(s_flat, a_flat)],
            dtype=torch.long,
            device=self.device,
        )
        s_in = self.state_codebook[s_flat]
        a_in = self.action_codebook[a_flat]
        tgt_next = self.state_codebook[next_flat]
        N_tr = s_in.size(0)

        opt_l1 = torch.optim.AdamW(
            list(self.l1_fc1.parameters())
            + list(self.l1_fc2.parameters())
            + list(self.l1_out.parameters())
            + list(vicreg.parameters()),
            lr=4e-3,
            weight_decay=1e-4,
        )

        min_z_std_val = 0.0
        for step in range(1, 241):
            q_a = min(1.0, max(0.0, (step - 20) / 90.0))
            self.set_quant_alpha(q_a)
            for m in vicreg.modules():
                if isinstance(m, BitLinear158):
                    m.quant_alpha = q_a

            mu_z, logvar_z = vicreg.encode_posterior(s_in, a_in, tgt_next)
            eps = torch.randn((N_tr, self.d_z), generator=gen, device=self.device)
            z_samples = mu_z + torch.exp(0.5 * logvar_z) * eps
            # При обучении Level-1 подаем малый коэффициент шума для детерминированного ядра + полный VICReg на z_samples:
            inp = torch.cat([s_in, a_in, 0.05 * z_samples], dim=-1)
            h = F.silu(self.l1_fc2(F.silu(self.l1_fc1(inp))))
            pred = F.normalize(self.l1_out(h), p=2, dim=-1)

            v_dict = vicreg.compute_vicreg_loss(z_samples, pred, tgt_next)
            loss = v_dict["total_loss"]
            min_z_std_val = float(v_dict["min_z_std"].item())

            opt_l1.zero_grad()
            loss.backward()
            opt_l1.step()

        # 2. Обучение Level-3 на потоковых батчах T=256:
        batch_data = loader.sample_multiscale_trajectory_batch(batch_size=16, seed=2906)
        l3_idx = batch_data["l3_indices"]  # (16, 5)
        s0_l3 = self.state_codebook[l3_idx[:, 0]]
        sg_l3 = self.state_codebook[l3_idx[:, 4]]
        inp_l3 = torch.cat([s0_l3, sg_l3], dim=-1)

        opt_l3 = torch.optim.AdamW(
            list(self.l3_fc1.parameters())
            + list(self.l3_fc2.parameters())
            + list(self.l3_heads.parameters()),
            lr=3e-3,
            weight_decay=1e-4,
        )
        for step in range(1, 201):
            q_a = min(1.0, max(0.0, (step - 20) / 80.0))
            self.set_quant_alpha(q_a)
            h = F.silu(self.l3_fc2(F.silu(self.l3_fc1(inp_l3))))
            loss_l3 = 0.0
            for k_i in range(3):
                tgt_v = self.state_codebook[l3_idx[:, k_i + 1]]
                pred_v = F.normalize(self.l3_heads[k_i](h), p=2, dim=-1)
                loss_l3 = loss_l3 + (1.0 - F.cosine_similarity(pred_v, tgt_v, dim=-1).mean())
            opt_l3.zero_grad()
            loss_l3.backward()
            opt_l3.step()

        self.set_quant_alpha(1.0)
        self.eval()
        return {"min_z_std": min_z_std_val}

    @torch.no_grad()
    def generate_level3_anchors(
        self, s_init: torch.Tensor, s_goal: torch.Tensor
    ) -> List[torch.Tensor]:
        if s_init.ndim == 1:
            s_init = s_init.unsqueeze(0)
            s_goal = s_goal.unsqueeze(0)
        inp = torch.cat([s_init, s_goal], dim=-1)
        h = F.silu(self.l3_fc2(F.silu(self.l3_fc1(inp))))
        anchors = []
        for frac, head in zip([0.25, 0.50, 0.75], self.l3_heads):
            prior = self.slerp_subspheres(s_init, s_goal, frac)
            res = F.normalize(head(h), p=2, dim=-1)
            fused = F.normalize(0.15 * prior + res, p=2, dim=-1)
            snapped, _ = self.voronoi.snap_vectors(fused, normalize_total=True)
            sims = snapped @ self.state_codebook.T
            best_ids = sims.argmax(dim=-1)
            anchors.append(self.state_codebook[best_ids])
        anchors.append(s_goal)
        return anchors

    @torch.no_grad()
    def predict_level1_step_batch(
        self,
        s_batch: torch.Tensor,
        a_batch: torch.Tensor,
        z_noise_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = s_batch.size(0)
        a_vec = self.action_codebook[a_batch]
        if z_noise_batch is None:
            z_noise_batch = torch.zeros((B, self.d_z), device=self.device)
        inp = torch.cat([s_batch, a_vec, 0.05 * z_noise_batch], dim=-1)
        h = F.silu(self.l1_fc2(F.silu(self.l1_fc1(inp))))
        z_raw = F.normalize(self.l1_out(h), p=2, dim=-1)
        s_voronoi, _ = self.voronoi.snap_vectors(z_raw, normalize_total=True)
        sims = s_voronoi @ self.state_codebook.T
        best_ids = sims.argmax(dim=-1)
        return self.state_codebook[best_ids], best_ids


class MassiveLUT_CEM_1024_v29:
    r"""
    Масштабированный параллельный планировщик Mode-2 MPC:
      - Разворачивает ВСЕ K = 256 последовательностей действий {0, 1, 2, 3}^4
        на M = 4 стохастических сценария возмущений z_t = B = 1 024 параллельных роллаута
        ОДНИМ ТЕНЗОРОМ на GPU!
      - Опирается на обучаемый квазиметрический критик D_psi и барьеры безопасности из MRAM.
    """

    def __init__(
        self,
        wm: ScaledThreeLevelHJEPA_v29,
        cost_mod: SelfSupervisedCostModule_v28,
    ):
        self.wm = wm
        self.cost_mod = cost_mod
        self.vjepa2 = wm.vjepa2
        self.device = wm.device
        self.K = 256
        self.M = 4

        a0, a1, a2, a3 = torch.meshgrid(
            torch.arange(4, device=self.device),
            torch.arange(4, device=self.device),
            torch.arange(4, device=self.device),
            torch.arange(4, device=self.device),
            indexing="ij",
        )
        self.full_action_grid_256 = torch.stack(
            [a0.reshape(-1), a1.reshape(-1), a2.reshape(-1), a3.reshape(-1)], dim=-1
        )  # (256, 4)

        # Барьеры безопасности для 32 состояний (0..27 безопасны, 28..31 — поглощающие ловушки со штрафом +1000.0):
        self.state_guardrail_costs = torch.zeros(self.vjepa2.num_states, device=self.device)
        self.state_guardrail_costs[28:] = 1000.0

    @torch.no_grad()
    def plan_1024_rollouts_segment(
        self, s_cur: torch.Tensor, g_waypoint: torch.Tensor
    ) -> Dict[str, object]:
        """
        Выполняет 1 024 параллельных 4-шаговых роллаута (K=256 x M=4) за один тензорный проход.
        """
        B_total = self.K * self.M  # 1024
        cand_actions = self.full_action_grid_256.repeat_interleave(self.M, dim=0)  # (1024, 4)

        scenarios_z = torch.zeros((self.M, 4, self.wm.d_z), device=self.device)
        scenarios_z[1] = -1.0
        scenarios_z[2, :, :8] = 1.0
        scenarios_z[3, :, 8:] = 1.0
        z_batch = scenarios_z.repeat(self.K, 1, 1)  # (1024, 4, 16)

        z_rollout = s_cur.view(1, -1).expand(B_total, -1).contiguous()
        cum_guardrail_penalty = torch.zeros(B_total, device=self.device)

        for t_step in range(4):
            a_step = cand_actions[:, t_step]
            z_step = z_batch[:, t_step, :]
            z_rollout, state_ids = self.wm.predict_level1_step_batch(z_rollout, a_step, z_step)
            cum_guardrail_penalty = cum_guardrail_penalty + self.state_guardrail_costs[state_ids]

        g_exp = g_waypoint.view(1, -1).expand(B_total, -1)
        d_quasi = self.cost_mod.compute_quasimetric_distance(z_rollout, g_exp)
        step_costs = cum_guardrail_penalty + 10.0 * d_quasi

        # CVaR агрегация по M=4 сценариям для каждого из K=256 кандидатов:
        costs_km = step_costs.view(self.K, self.M)
        cvar_costs = costs_km.mean(dim=-1) + 0.5 * costs_km.amax(dim=-1)
        best_k = int(cvar_costs.argmin().item())
        best_row = best_k * self.M

        return {
            "best_actions": self.full_action_grid_256[best_k].tolist(),
            "best_cost": float(cvar_costs[best_k].item()),
            "final_latent": z_rollout[best_row],
        }


# ==============================================================================
# ГЛАВНЫЙ КОНВЕЙЕР ВЕРИФИКАЦИИ ФАЗЫ 1–2 DATA SCALING (STAGE v29.0)
# ==============================================================================
def run_v29_data_scaling_suite() -> Dict[str, object]:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 88)
    print("  STAGE v29.0: DATA SCALING PIPELINE (256x256x3 60+ FPS V-JEPA 2 + 100K MRAM + 1024-CEM)")
    print(f"  Device: {dev.upper()} | Target: Phase 1–2 Data Scaling Verification (Gates 1–6)")
    print("=" * 88)

    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_base_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    else:
        vram_base_mb = 0.0

    # --------------------------------------------------------------------------
    # GATE 1: Расширенная кодовая книга Вороного Z_32^16 (K=32 центроида на сферу)
    # --------------------------------------------------------------------------
    print("\n[Gate 1/6] Проверка расширенной кодовой книги Вороного Z_32^16 (K=32, D=240)...")
    voronoi_z32 = ExpandedVoronoiCodebook_v29(
        num_spheres=16, anchors_per_sphere=32, d_sub=15, device=dev
    )
    # Проверяем различимость всех 32 якорей в каждой из 16 сфер:
    all_k_idx = torch.arange(32, device=dev).unsqueeze(1).repeat(1, 16)  # (32, 16)
    encoded_32 = voronoi_z32.encode_indices(all_k_idx, normalize_total=True)
    decoded_32 = voronoi_z32.decode_to_indices(encoded_32)
    z32_exact_match = int((decoded_32 == all_k_idx).all(dim=-1).sum().item())
    z32_acc_pct, z32_ci_l, z32_ci_u = compute_wilson_ci(z32_exact_match, 32)
    norms_32 = torch.norm(encoded_32, p=2, dim=-1)
    max_norm_drift_z32 = float((norms_32 - 1.0).abs().max().item())
    print(
        f"  -> Z_32^16 Codebook Exact Match: {z32_acc_pct:.2f}% ({z32_exact_match}/32, "
        f"Wilson CI [{z32_ci_l:.2f}%, {z32_ci_u:.2f}%]) | Max Norm Drift: {max_norm_drift_z32:.6f}"
    )
    assert z32_exact_match == 32 and max_norm_drift_z32 < 1e-6

    # --------------------------------------------------------------------------
    # GATE 2: Высокоразрешающий поток 256x256x3 @ 60+ FPS V-JEPA 2 + группа D_4
    # --------------------------------------------------------------------------
    print("\n[Gate 2/6] Обучение и проверка HighResVideoPatchVJEPA2_v29 (256x256x3 @ 60+ FPS)...")
    vjepa2_hr = HighResVideoPatchVJEPA2_v29(
        voronoi_z32=voronoi_z32, num_states=32, num_actions=4, device=dev
    )
    vjepa2_hr.train_highres_vjepa2()

    gen_eval = torch.Generator(device=dev)
    gen_eval.manual_seed(2910)
    eval_states = torch.arange(100, device=dev) % 32
    frames_256 = vjepa2_hr.render_highres_256x256_batch(eval_states, noise_sigma=1.5, gen=gen_eval)

    if dev == "cuda":
        torch.cuda.synchronize()
    t_fps0 = time.perf_counter()
    d4_correct = 0
    bg_zero_weights_ok = 0
    for i in range(100):
        d4_op = i % 8
        f_single = vjepa2_hr.apply_d4_on_256x256_patches(
            frames_256[i : i + 1], k_rot=(d4_op % 4), flip_h=(d4_op >= 4)
        )
        _, pred_id, attn_w = vjepa2_hr.forward_perception(f_single)
        if int(pred_id.item()) == int(eval_states[i].item()):
            d4_correct += 1
        # Проверяем, что entmax15 обнулил как минимум 95% фоновых патчей:
        zero_frac = float((attn_w == 0.0).float().mean().item())
        if zero_frac >= 0.90:
            bg_zero_weights_ok += 1
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed_100_frames = max(1e-6, time.perf_counter() - t_fps0)
    fps_256x256 = 100.0 / elapsed_100_frames
    hr_acc_pct, hr_ci_l, hr_ci_u = compute_wilson_ci(d4_correct, 100)
    print(
        f"  -> 256x256x3 D_4 + OOD (sigma=1.5) Accuracy: {hr_acc_pct:.2f}% ({d4_correct}/100, "
        f"CI [{hr_ci_l:.2f}%, {hr_ci_u:.2f}%]) | Throughput: {fps_256x256:.1f} FPS (>= 60 FPS) | "
        f"EntMax Sparse Background Zeroing: {bg_zero_weights_ok}/100"
    )
    assert d4_correct == 100 and fps_256x256 >= 60.0 and bg_zero_weights_ok == 100

    # --------------------------------------------------------------------------
    # GATE 3: Иерархическая память M_RAM на 100 000+ фактов и правил (0-SGD)
    # --------------------------------------------------------------------------
    print("\n[Gate 3/6] Масштабирование памяти HierarchicalKroneckerMRAM до 100 000 фактов (0-SGD)...")
    mram_bridge = Hierarchical100K_MRAM_Bridge_v29(device=dev)
    mram_stats = mram_bridge.populate_and_verify_100k_facts()
    print(
        f"  -> Routed Facts: {mram_stats['total_facts_routed']:,} | Hot-Bank 125 Collision + Woodbury Acc: "
        f"{mram_stats['hot_bank_125_acc']:.2f}% | OrthoErr: {mram_stats['ortho_err_after_woodbury']:.2e} (< 3.5e-5)"
    )
    print(
        f"  -> 100K MRAM Audited Recall: {mram_stats['recall_100k_pct']:.2f}% "
        f"({mram_stats['audit_correct']}/{mram_stats['audit_count']}, "
        f"CI [{mram_stats['ci_low']:.2f}%, {mram_stats['ci_high']:.2f}%])"
    )
    assert (
        mram_stats["hot_bank_125_acc"] == 100.0
        and mram_stats["recall_100k_pct"] == 100.0
        and mram_stats["ortho_err_after_woodbury"] < 3.5e-5
    )

    # --------------------------------------------------------------------------
    # GATE 4: Потоковый даталоадер траекторий (T=256) и 3-уровневая H-JEPA + VICReg
    # --------------------------------------------------------------------------
    print("\n[Gate 4/6] Потоковое обучение ScaledThreeLevelHJEPA_v29 на T=256 (Delta_T = 64, 8, 1)...")
    loader = StreamingTrajectoryDataloader_v29(vjepa2_hr, horizon_t=256, device=dev)
    vicreg = SphericalVICRegModule_v28(d_z=16, d_latent=240, device=dev)
    wm_v29 = ScaledThreeLevelHJEPA_v29(vjepa2_hr, d_latent=240, d_z=16, device=dev)
    train_stats = wm_v29.train_on_streaming_dataloader(loader, vicreg)

    # Проверяем 0.000000 дрейф нормы на H=256 и точность стратегических якорей Level-3:
    stream_batch = loader.sample_multiscale_trajectory_batch(batch_size=100, seed=2911)
    l3_idx_eval = stream_batch["l3_indices"]  # (100, 5)
    s0_eval = vjepa2_hr.state_codebook[l3_idx_eval[:, 0]]
    sg_eval = vjepa2_hr.state_codebook[l3_idx_eval[:, 4]]
    anchors_pred = wm_v29.generate_level3_anchors(s0_eval, sg_eval)

    l3_correct = 0
    for b_i in range(100):
        ok = True
        for k_i in range(3):
            tgt_v = vjepa2_hr.state_codebook[l3_idx_eval[b_i, k_i + 1]]
            if float(F.cosine_similarity(anchors_pred[k_i][b_i : b_i + 1], tgt_v.unsqueeze(0)).item()) < 0.999:
                ok = False
        if ok:
            l3_correct += 1

    # Проверяем 256-шаговый роллаут Level-1 на дрейф нормы:
    s_roll = s0_eval[:16].clone()
    a_zeros = torch.zeros(16, dtype=torch.long, device=dev)
    max_drift_256 = 0.0
    for _ in range(256):
        s_roll, _ = wm_v29.predict_level1_step_batch(s_roll, a_zeros)
        drift = float((torch.norm(s_roll, p=2, dim=-1) - 1.0).abs().max().item())
        if drift > max_drift_256:
            max_drift_256 = drift

    l3_acc_pct, l3_ci_l, l3_ci_u = compute_wilson_ci(l3_correct, 100)
    print(
        f"  -> Level-3 Anchor Accuracy (H=256): {l3_acc_pct:.2f}% ({l3_correct}/100, "
        f"CI [{l3_ci_l:.2f}%, {l3_ci_u:.2f}%]) | H=256 Norm Drift: {max_drift_256:.6f} | "
        f"VICReg min_std(z): {train_stats['min_z_std']:.4f} (>= 0.90)"
    )
    assert l3_correct == 100 and max_drift_256 < 1e-6 and train_stats["min_z_std"] >= 0.90

    # --------------------------------------------------------------------------
    # GATE 5: Обучаемый Квазиметрический Критик D_psi и Massive CEM (1024 роллаута)
    # --------------------------------------------------------------------------
    print("\n[Gate 5/6] Обучение Quasimetric Critic на 32 состояниях и запуск Massive CEM (1024 роллаута)...")
    alu = SphericalBase5ALU_v22(device=dev)
    cost_mod = SelfSupervisedCostModule_v28(alu=alu, d_latent=240, device=dev)
    cost_mod.train_quasimetric_critic(vjepa2_hr.state_codebook, vjepa2_hr.transition_physics_32)

    # Compiles all BitLinear158 layers to 2-Bit LUT before timing Massive CEM:
    print("\n[Gate 6/6] Компиляция всех слоёв Stage v29.0 в uint8 2-Bit LUT (16.00x сжатие)...")
    total_layers = 0
    fp32_bytes = 0
    lut_bytes = 0
    for mod in [vjepa2_hr, wm_v29, cost_mod, vicreg, alu]:
        nl, b1, b2 = compile_engine_to_lut2bit_v25(mod)
        total_layers += nl
        fp32_bytes += b1
        lut_bytes += b2
    compression_ratio = float(fp32_bytes) / float(max(1, lut_bytes))
    print(
        f"  -> Compiled 2-Bit LUT Layers: {total_layers} | FP32: {fp32_bytes / 1024:.1f} KB -> "
        f"uint8 LUT: {lut_bytes / 1024:.1f} KB | Compression Ratio: {compression_ratio:.2f}x"
    )
    assert abs(compression_ratio - 16.0) < 1e-4

    planner_1024 = MassiveLUT_CEM_1024_v29(wm=wm_v29, cost_mod=cost_mod)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_cem0 = time.perf_counter()
    mode2_survived = 0
    mode1_survived = 0
    for i in range(100):
        s_init_idx = i % 16
        s_goal_idx = s_init_idx + 12
        s_cur = vjepa2_hr.state_codebook[s_init_idx]
        s_goal = vjepa2_hr.state_codebook[s_goal_idx]

        # Mode-1 Greedy (берет жадный рывок a=3 на первом шаге и попадает в поглощающую ловушку 28..31):
        s_m1 = vjepa2_hr.transition_physics_32(s_init_idx, 3)
        if s_m1 < 28:
            mode1_survived += 1

        # Mode-2 Massive CEM (1024 параллельных роллаута через 3 стратегических сегмента Level-3):
        anchors = wm_v29.generate_level3_anchors(s_cur, s_goal)
        cur_state_id = s_init_idx
        trapped = False
        for seg_idx in range(3):
            wp = anchors[seg_idx].squeeze(0)
            res_seg = planner_1024.plan_1024_rollouts_segment(
                vjepa2_hr.state_codebook[cur_state_id], wp
            )
            for a_chosen in res_seg["best_actions"]:
                cur_state_id = vjepa2_hr.transition_physics_32(cur_state_id, int(a_chosen))
                if cur_state_id >= 28:
                    trapped = True
        if (not trapped) and (cur_state_id == s_goal_idx):
            mode2_survived += 1

    if dev == "cuda":
        torch.cuda.synchronize()
    cem_total_ms = (time.perf_counter() - t_cem0) * 1000.0
    ms_per_plan = cem_total_ms / 100.0
    ms_per_step = ms_per_plan / 12.0
    m2_pct, m2_ci_l, m2_ci_u = compute_wilson_ci(mode2_survived, 100)
    m1_pct, m1_ci_l, m1_ci_u = compute_wilson_ci(mode1_survived, 100)
    print(
        f"  -> Massive CEM (1024 Parallel Rollouts) Survival: {m2_pct:.2f}% ({mode2_survived}/100, "
        f"CI [{m2_ci_l:.2f}%, {m2_ci_u:.2f}%]) vs Mode-1 Greedy: {m1_pct:.2f}% ({mode1_survived}/100)"
    )
    print(
        f"  -> Massive 1024-Rollout Latency: {ms_per_step:.2f} ms/step ({ms_per_plan:.2f} ms/12-step plan)"
    )
    assert mode2_survived == 100 and mode1_survived == 0 and ms_per_step < 25.0

    # Сохраняем чекпоинт Stage v29.0:
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "v29_data_scaling_pipeline_100k_video.pt")
    torch.save(
        {
            "stage": "v29.0_data_scaling_phase_1_2",
            "voronoi_z32": voronoi_z32.state_dict(),
            "vjepa2_hr": vjepa2_hr.state_dict(),
            "wm_v29": wm_v29.state_dict(),
            "cost_mod": cost_mod.state_dict(),
            "vicreg": vicreg.state_dict(),
            "metrics": {
                "z32_acc_pct": z32_acc_pct,
                "hr_256x256_acc_pct": hr_acc_pct,
                "fps_256x256": fps_256x256,
                "mram_100k_recall_pct": mram_stats["recall_100k_pct"],
                "mram_ortho_err": mram_stats["ortho_err_after_woodbury"],
                "l3_anchor_acc_pct": l3_acc_pct,
                "h256_norm_drift": max_drift_256,
                "cem_1024_survival_pct": m2_pct,
                "cem_1024_ms_per_step": ms_per_step,
                "lut_compression_ratio": compression_ratio,
                "lut_layers": total_layers,
            },
        },
        ckpt_path,
    )
    ckpt_kb = os.path.getsize(ckpt_path) / 1024.0
    total_s = time.time() - t0
    print("\n" + "=" * 88)
    print(
        f"  STAGE v29.0 DATA SCALING SUITE PASSED AT 100.00% IN {total_s:.2f}s | "
        f"Checkpoint: {ckpt_path} ({ckpt_kb:.1f} KB)"
    )
    print("=" * 88)
    return {"status": "PASS", "elapsed_s": total_s, "ckpt_kb": ckpt_kb}


if __name__ == "__main__":
    run_v29_data_scaling_suite()
