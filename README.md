# pairwise-alpha-matting-trimap

## 领域

游戏、图形与多媒体

## 主题

抠图三色带与 alpha 求解

## 说明

梯度自适应带、块间融合与多尺度顺序。

初始状态下 `imaging/alpha_matting_trimap.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复点与实现对照

| 缺陷 | 修复 | 实现位置 |
| --- | --- | --- |
| 三色带固定像素膨胀，发丝等细结构被划进确定前景、alpha 变硬 | 未知带半径随局部梯度自适应（99 分位归一化，平坦区取基础带宽、强梯度区扩到 `max_width`），发丝整体落入未知区 | `adaptive_band` / `trimap_build` |
| 未知区分块边界处邻域样本不足 | 采样窗口沿块四边外扩一个采样半径，图像边界截断 | `sample_border` |
| 大图分块并行求解时块间 alpha 不连续 | 重叠区按到块边界的距离权重加权融合（块心权重 1、边界衰减到 0） | `block_fusion` / `matte_solve` |
| 异步多尺度传播粗尺度晚于细尺度写回 | 粗 → 细层级顺序提交，`MultiScaleSolver` 对乱序写回直接抛错 | `scale_order` / `MultiScaleSolver` |
| trimap 复用上一张图缓存、前景不随新图重算 | 按图像内容指纹校验缓存，换图/换掩码必然失效并逐图重算 | `trimap_recompute` |
| 边缘 alpha 变硬无校验 | 边缘带 alpha 直方图校验：硬切 0/1 比例与软像素占比双阈值 | `edge_check` |

其他配套能力：纯 numpy 欧氏距离变换（`edt`）、分块闭式 alpha 求解（`matte_solve`，
采样半径按 `min(H,W)` 缩放）、未知带内精修（`alpha_refine`）、指标（`metrics`）、
入参校验（`validation`）、审计（`audit`）、失败兜底（`fallback`）、
逐位确定性校验（`determinism`）。

测试：

- `tests/test_alpha_matting_trimap.py`：原始契约测试（15 个函数标量调用返回非 None）。
- `tests/test_fixes_behavior.py`：六个修复点的行为测试，覆盖发丝保留、边界外扩、
  块接缝连续性、多尺度乱序拒绝、缓存失效、边缘直方图检出及端到端软过渡。

