"""六个修复点的行为测试：不只验证返回非 None，而是验证缺陷确实被修复。"""

import numpy as np
import pytest

import imaging.alpha_matting_trimap as mod


# --------------------------------------------------------------------------
# 修复点 1：三色带按梯度自适应；发丝等细结构落入未知区而非确定前景
# --------------------------------------------------------------------------
def test_adaptive_band_widens_at_strong_gradient():
    h, w = 40, 40
    gradient = np.zeros((h, w))
    gradient[:, w // 2:] = 100.0  # 右半为强梯度（边缘/发丝）
    widths = mod.adaptive_band(gradient, base_width=3.0, max_width=12.0)

    assert widths.shape == (h, w)
    assert np.allclose(widths[:, : w // 2 - 1], 3.0)          # 平坦区 = 基础带宽
    assert np.all(widths[:, w // 2 + 1:] >= 11.0)             # 强梯度区显著加宽
    assert np.all(widths[:, w // 2 + 1:] > widths[:, : w // 2 - 1])


def test_trimap_keeps_thin_hair_in_unknown_region():
    h, w = 60, 80
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :30] = True                       # 左侧实心前景
    mask[30, 30:56] = True                    # 1px 发丝向右延伸
    image = np.where(mask, 255.0, 0.0)        # 前景白、背景黑 -> 发丝处梯度极强

    trimap = mod.trimap_build(mask, image, base_width=3.0, max_width=12.0)

    hair = trimap[30, 30:56]
    assert np.all(hair == mod.TRIMAP_UNKNOWN), "发丝细结构必须落入未知区，不能被划进确定前景"
    # 实心前景内部仍保持确定前景，自适应带宽不会把整图都变成未知区
    assert trimap[2, 2] == mod.TRIMAP_FG
    assert mod.TRIMAP_BG in np.unique(trimap)
    assert mod.unknown_region(trimap).sum() > 0


# --------------------------------------------------------------------------
# 修复点 2：未知区采样带沿分块边界外扩，图像边界处截断
# --------------------------------------------------------------------------
def test_sample_border_extends_beyond_block_and_clamps_at_image_edge():
    trimap = np.full((100, 100), mod.TRIMAP_BG, dtype=np.uint8)
    trimap[40:60, 40:60] = mod.TRIMAP_UNKNOWN

    geo = mod.sample_border(trimap, radius=10, block_box=(30, 30, 60, 60))
    assert geo["window"] == (20, 20, 70, 70)
    assert geo["expansion"] == (10, 10, 10, 10)
    # 外扩窗口必须包住块外的已知样本点
    assert geo["fg_points"].shape[1] == 2 or geo["bg_points"].shape[1] == 2

    geo_edge = mod.sample_border(trimap, radius=10, block_box=(0, 0, 30, 30))
    assert geo_edge["window"] == (0, 0, 40, 40)
    assert geo_edge["expansion"] == (0, 0, 10, 10)   # 越界部分被截断


# --------------------------------------------------------------------------
# 修复点 3：分块重叠区加权融合，块间 alpha 连续
# --------------------------------------------------------------------------
def test_block_fusion_blends_overlap_without_hard_seam():
    h, w = 64, 64
    tile_a = np.full((64, 48), 0.2)
    tile_b = np.full((64, 48), 0.8)
    tiles = [((0, 0, 64, 48), tile_a), ((0, 16, 64, 64), tile_b)]

    fused = mod.block_fusion(tiles, shape=(h, w), overlap=32)

    assert fused.shape == (h, w)
    assert np.all(np.isfinite(fused))
    assert fused.min() >= 0.0 and fused.max() <= 1.0
    # 重叠区 (16..48) 内的像素必须是两块的加权混合，而非取某一块的硬值
    overlap_vals = fused[:, 16:48]
    assert np.all((overlap_vals > 0.2) & (overlap_vals < 0.8))
    # 融合后任意水平相邻像素的跳变远小于原始块差 0.6（无硬接缝）
    max_jump = np.max(np.abs(np.diff(fused, axis=1)))
    assert max_jump < 0.3


def test_matte_solve_tiled_output_is_continuous_and_constrained():
    rng = np.random.default_rng(7)
    h, w = 300, 320
    mask = np.zeros((h, w), dtype=bool)
    mask[:, : w // 2] = True
    image = np.where(mask, 255.0, 0.0)[..., np.newaxis].repeat(3, axis=2)
    image += rng.normal(0, 2.0, image.shape)
    trimap = mod.trimap_build(mask, image, base_width=4.0)

    alpha = mod.matte_solve(image, trimap, block_size=96, overlap=24)

    assert alpha.shape == (h, w)
    assert np.all(np.isfinite(alpha))
    assert np.allclose(alpha[trimap == mod.TRIMAP_FG], 1.0)
    assert np.allclose(alpha[trimap == mod.TRIMAP_BG], 0.0)
    # 分块边界（96、192 列附近）不允许出现硬接缝；图像内容自身的边缘（160 列）不参与度量
    block_seam_jumps = [
        np.max(np.abs(np.diff(alpha[:, c - 1:c + 1], axis=1)))
        for c in (96, 192)
    ]
    assert max(block_seam_jumps) < 0.25


# --------------------------------------------------------------------------
# 修复点 4：多尺度按粗 -> 细层级顺序提交，拒绝乱序写回
# --------------------------------------------------------------------------
def test_scale_order_is_coarse_to_fine():
    plan = mod.scale_order(3)
    assert [p["scale"] for p in plan] == [0.25, 0.5, 1.0]
    assert [p["level"] for p in plan] == [0, 1, 2]

    plan2 = mod.scale_order([1.0, 0.25, 0.5])
    assert [p["scale"] for p in plan2] == [0.25, 0.5, 1.0]

    with pytest.raises(ValueError):
        mod.scale_order(0)
    with pytest.raises(ValueError):
        mod.scale_order([0.5, 0.5])


def test_multiscale_solver_rejects_out_of_order_writeback():
    solver = mod.MultiScaleSolver(3)
    solver.submit(0, {"alpha": "coarse"})
    with pytest.raises(RuntimeError):
        solver.submit(2, {"alpha": "fine-before-mid"})   # 粗尺度未完成即写细尺度
    solver.submit(1, {"alpha": "mid"})
    solver.submit(2, {"alpha": "fine"})
    assert solver.finished


# --------------------------------------------------------------------------
# 修复点 5：trimap 逐图重算，换图后缓存必然失效
# --------------------------------------------------------------------------
def test_trimap_recompute_invalidates_cache_on_new_image():
    mask_a = np.zeros((40, 40), dtype=bool)
    mask_a[:, :20] = True
    image_a = np.where(mask_a, 255.0, 0.0)

    first = mod.trimap_recompute(mask_a, image_a)
    assert first["recomputed"] is True and first["cache_hit"] is False

    second = mod.trimap_recompute(mask_a, image_a, cache=first["cache"])
    assert second["cache_hit"] is True and second["recomputed"] is False
    assert np.array_equal(second["trimap"], first["trimap"])

    # 换一张图（同尺寸、不同内容）：旧缓存不得复用
    image_b = image_a.copy()
    image_b[10, 10] = 1.0
    third = mod.trimap_recompute(mask_a, image_b, cache=first["cache"])
    assert third["cache_hit"] is False and third["recomputed"] is True

    # 换掩码（前景区域变化）同样失效
    mask_b = mask_a.copy()
    mask_b[5, 5] = False
    fourth = mod.trimap_recompute(mask_b, image_a, cache=first["cache"])
    assert fourth["cache_hit"] is False


# --------------------------------------------------------------------------
# 修复点 6：边缘按 alpha 直方图校验，硬边被检出
# --------------------------------------------------------------------------
def test_edge_check_detects_hardened_alpha_and_accepts_soft_alpha():
    trimap = np.full((20, 20), mod.TRIMAP_UNKNOWN, dtype=np.uint8)
    trimap[:5, :] = mod.TRIMAP_FG
    trimap[-5:, :] = mod.TRIMAP_BG

    hard_alpha = np.where(trimap == mod.TRIMAP_FG, 1.0, 0.0)   # 未知区全被硬切成 0/1
    report_hard = mod.edge_check(hard_alpha, trimap)
    assert report_hard["ok"] is False
    assert report_hard["hard_ratio"] == 1.0
    assert any("变硬" in r for r in report_hard["reasons"])
    assert report_hard["histogram"].sum() == report_hard["edge_pixels"]

    soft_alpha = hard_alpha.copy()
    band = trimap == mod.TRIMAP_UNKNOWN
    soft_alpha[band] = np.linspace(0.2, 0.8, band.sum())
    report_soft = mod.edge_check(soft_alpha, trimap)
    assert report_soft["ok"] is True
    assert report_soft["hard_ratio"] == 0.0
    assert report_soft["soft_ratio"] > 0.5

    bad_alpha = soft_alpha.copy()
    bad_alpha[0, 0] = np.nan
    assert mod.edge_check(bad_alpha, trimap)["ok"] is False


# --------------------------------------------------------------------------
# 配套能力：确定性、校验、审计、兜底、指标
# --------------------------------------------------------------------------
def test_determinism_repeated_solves_are_bit_identical():
    mask = np.zeros((50, 60), dtype=bool)
    mask[:, :30] = True
    image = np.where(mask, 200.0, 30.0)[..., np.newaxis].repeat(3, axis=2)
    trimap = mod.trimap_build(mask, image)

    report = mod.determinism(mod.matte_solve, image, trimap, block_size=32, overlap=8)
    assert report["deterministic"] is True
    assert report["max_abs_diff"] == 0.0


def test_validation_audit_metrics_fallback_and_refine():
    mask = np.zeros((40, 40), dtype=bool)
    mask[:20, :] = True
    image = np.where(mask, 255.0, 0.0)
    trimap = mod.trimap_build(mask, image)
    alpha = mod.matte_solve(image[..., np.newaxis], trimap, block_size=32)

    assert mod.validation(image, trimap, alpha)["ok"] is True
    bad_trimap = trimap.copy()
    bad_trimap[0, 0] = 7
    assert mod.validation(image, bad_trimap)["ok"] is False

    audit = mod.audit(alpha, trimap)
    assert audit["n_unknown"] > 0 and audit["finite"] is True
    assert "edge" in audit and "hard_ratio" in audit["edge"]

    gt = np.where(mask, 1.0, 0.0)
    m = mod.metrics(alpha, gt_alpha=gt, trimap=trimap)
    assert 0.0 <= m["mae"] <= 1.0 and m["sad"] >= 0.0 and m["edge_mae"] is not None

    refined = mod.alpha_refine(alpha, trimap, passes=2)
    assert np.allclose(refined[trimap == mod.TRIMAP_FG], 1.0)
    assert np.allclose(refined[trimap == mod.TRIMAP_BG], 0.0)

    fb = mod.fallback(image=image, mask=mask)
    assert fb["degraded"] is True and fb["alpha"].shape == mask.shape
    assert set(np.unique(fb["trimap"]).tolist()).issubset(
        {mod.TRIMAP_BG, mod.TRIMAP_FG, mod.TRIMAP_UNKNOWN}
    )


def test_end_to_end_pipeline_soft_transition_on_synthetic_edge():
    """端到端：白前景/黑背景合成图，未知带 alpha 应存在 0~1 之间的软过渡并通过边缘校验。"""
    h, w = 80, 80
    mask = np.zeros((h, w), dtype=bool)
    mask[:, : w // 2] = True
    image = np.where(mask, 255.0, 0.0)[..., np.newaxis].repeat(3, axis=2)

    trimap = mod.trimap_build(mask, image, base_width=4.0, max_width=10.0)
    alpha = mod.matte_solve(image, trimap, block_size=48, overlap=12)
    alpha = mod.alpha_refine(alpha, trimap)

    check = mod.edge_check(alpha, trimap)
    assert check["ok"] is True, check["reasons"]
    band_values = alpha[trimap == mod.TRIMAP_UNKNOWN]
    assert np.any((band_values > 0.05) & (band_values < 0.95))
