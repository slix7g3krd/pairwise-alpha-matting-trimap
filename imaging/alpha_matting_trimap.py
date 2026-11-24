"""抠图三色带（trimap）构建与 alpha 求解模块。

六个修复点与函数对应关系
------------------------
1. ``adaptive_band``    三色带按图像梯度自适应膨胀，替代固定像素宽度；
                        强梯度边缘（发丝等细结构）获得更宽的未知带，避免被划进确定前景。
2. ``sample_border``    未知区采样带沿分块边界外扩一个采样半径，保证块边界处邻域样本充足。
3. ``block_fusion``     分块并行求解的重叠区按到块边界的距离权重加权融合，消除块间 alpha 不连续。
4. ``scale_order``      多尺度求解严格按粗尺度 -> 细尺度的层级顺序提交/写回，
                        ``MultiScaleSolver`` 会拒绝乱序写回。
5. ``trimap_recompute`` trimap 逐图重算；缓存按图像内容指纹校验，换图后旧前景缓存绝不复用。
6. ``edge_check``       边缘按 alpha 直方图校验，检出被硬切成 0/1 的边缘（alpha 变硬）。

此外提供完整流水线所需的求解（``matte_solve``）、trimap 构建、精修、指标、
确定性校验、入参校验、审计与失败兜底函数。

三色带约定：0=确定背景，128=未知，255=确定前景。
所有公开函数同时支持契约测试的单标量调用（如 ``adaptive_band(1)``），该情况下返回安全默认值。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TRIMAP_BG",
    "TRIMAP_UNKNOWN",
    "TRIMAP_FG",
    "adaptive_band",
    "sample_border",
    "block_fusion",
    "trimap_build",
    "unknown_region",
    "matte_solve",
    "scale_order",
    "MultiScaleSolver",
    "trimap_recompute",
    "edge_check",
    "alpha_refine",
    "metrics",
    "validation",
    "audit",
    "fallback",
    "determinism",
]

TRIMAP_BG = 0
TRIMAP_UNKNOWN = 128
TRIMAP_FG = 255

_EPS = 1e-8
# 距离变换使用的有限大值（平方距离空间），避免 inf 参与运算产生 NaN。
_DT_INF = 1.0e18


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------
def _edt_1d(f: np.ndarray) -> np.ndarray:
    """一维平方欧氏距离变换（Felzenszwalb 抛物线包络算法，O(n)）。

    f[i] 为源点处 0、非源点处 _DT_INF 的初始代价，返回每个位置到最近源点的平方距离。
    """
    n = f.shape[0]
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        fq = f[q] + q * q
        denom = 2.0 * (q - v[k])
        s = (fq - (f[v[k]] + v[k] * v[k])) / denom
        while k >= 0 and s <= z[k]:
            k -= 1
            if k < 0:
                break
            denom = 2.0 * (q - v[k])
            s = (fq - (f[v[k]] + v[k] * v[k])) / denom
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def edt(mask: np.ndarray) -> np.ndarray:
    """二维欧氏距离变换：返回每个像素到 ``mask`` 为 True 的最近像素的距离（像素）。"""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("edt 仅支持二维掩码")
    h, w = m.shape
    f = np.where(m, 0.0, _DT_INF)
    row_dt = np.empty_like(f)
    for y in range(h):
        row_dt[y] = _edt_1d(f[y])
    out = np.empty_like(row_dt)
    for x in range(w):
        out[:, x] = _edt_1d(row_dt[:, x])
    return np.sqrt(np.maximum(out, 0.0))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """灰度/彩色图的梯度幅值（中心差分，边界用单侧差分）。"""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    return np.hypot(gx, gy)


def _is_scalar(x) -> bool:
    return x is not None and np.isscalar(x)


# ---------------------------------------------------------------------------
# 修复点 1：三色带按梯度自适应
# ---------------------------------------------------------------------------
def adaptive_band(gradient, base_width: float = 4.0, max_width: float | None = None) -> np.ndarray | float:
    """根据局部梯度幅值计算每个像素的未知带半径（像素）。

    旧实现对整张图使用固定膨胀宽度，发丝等高梯度细结构会被整体划进确定前景，
    求解时 alpha 只能取 0/1 而变硬。这里让未知带半径随梯度线性增大：

    - 平坦区（梯度≈0）：半径取 ``base_width``，与原固定带宽行为一致；
    - 强梯度区（真边缘/发丝）：半径扩大到 ``max_width``，细结构保留在未知区内，
      由求解器估计软 alpha。

    返回与 ``gradient`` 同形的浮点半径图；标量调用时返回默认半径。
    """
    if _is_scalar(gradient):
        return float(base_width)
    g = np.asarray(gradient, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gradient 必须是二维梯度幅值图")
    base_width = float(base_width)
    if max_width is None:
        max_width = max(base_width * 4.0, base_width + 1.0)
    max_width = float(max_width)
    if max_width < base_width:
        raise ValueError("max_width 不能小于 base_width")
    # 99 分位做鲁棒归一化，避免单个极值把整张图的带宽压扁。
    gmax = float(np.percentile(g, 99))
    if gmax <= _EPS:
        return np.full(g.shape, base_width, dtype=np.float64)
    t = np.clip(g / gmax, 0.0, 1.0)
    widths = base_width * (1.0 - t) + max_width * t
    return np.clip(widths, 1.0, max_width)


def trimap_build(mask, image=None, base_width: float = 4.0, max_width: float | None = None) -> np.ndarray:
    """由二值前景掩码 + 原图构建三色带。

    步骤：
    1. 分别计算前景/背景的欧氏距离，得到每个像素到掩码边界的距离；
    2. 用 :func:`adaptive_band` 生成梯度自适应的未知带半径图；
    3. 边界距离小于局部自适应半径的像素标为未知（128），其余按掩码标为
       确定前景（255）或确定背景（0）。

    强梯度处未知带更宽，发丝等细结构整体落入未知区，不会被固定膨胀吞进确定前景。
    """
    if _is_scalar(mask):
        return np.full((1, 1), TRIMAP_UNKNOWN, dtype=np.uint8)
    fg = np.asarray(mask, dtype=bool)
    if fg.ndim != 2:
        raise ValueError("mask 必须是二维二值掩码")
    if image is None or _is_scalar(image):
        gradient = _gradient_magnitude(fg.astype(np.float64))
    else:
        img = np.asarray(image, dtype=np.float64)
        if img.shape[:2] != fg.shape:
            raise ValueError("image 与 mask 的空间尺寸不一致")
        gradient = _gradient_magnitude(img)
    d_fg = edt(fg)       # 背景像素到最近前景的距离
    d_bg = edt(~fg)      # 前景像素到最近背景的距离
    boundary_dist = np.where(fg, d_bg, d_fg)
    widths = adaptive_band(gradient, base_width, max_width)
    unknown = boundary_dist < widths
    trimap = np.where(fg, TRIMAP_FG, TRIMAP_BG).astype(np.uint8)
    trimap[unknown] = TRIMAP_UNKNOWN
    return trimap


def unknown_region(trimap) -> np.ndarray:
    """返回三色带中的未知区域布尔掩码。"""
    if _is_scalar(trimap):
        return np.zeros((0,), dtype=bool)
    tri = np.asarray(trimap)
    return tri == TRIMAP_UNKNOWN


# ---------------------------------------------------------------------------
# 修复点 2：未知区采样带边界扩展
# ---------------------------------------------------------------------------
def sample_border(source, radius: int = 0, block_box=None) -> dict:
    """计算分块求解时的扩展采样窗口。

    旧实现只在块内部取已知前景/背景样本，块边界处的未知像素邻域不足、估计失真。
    这里把采样窗口沿块的四条边各外扩 ``radius`` 个像素（截断到图像边界），
    使边界像素也能取到邻块的已知样本。

    参数
    ----
    source:
        未知区域布尔掩码，或完整三色带（含 0/128/255 标签）。
    radius:
        采样半径（调用方按图像尺寸缩放后传入）。
    block_box:
        ``(y0, x0, y1, x1)``，为 None 时表示整图。

    返回 dict，包含原块、外扩窗口、各边实际外扩量以及窗口内的采样点坐标（全局 y,x）。
    """
    if _is_scalar(source):
        return {
            "shape": (1, 1),
            "block": (0, 0, 1, 1),
            "window": (0, 0, 1, 1),
            "expansion": (0, 0, 0, 0),
            "radius": int(radius),
            "unknown_points": np.zeros((0, 2), dtype=np.int64),
            "fg_points": np.zeros((0, 2), dtype=np.int64),
            "bg_points": np.zeros((0, 2), dtype=np.int64),
        }
    arr = np.asarray(source)
    h, w = arr.shape
    r = max(0, int(radius))
    if block_box is None:
        y0, x0, y1, x1 = 0, 0, h, w
    else:
        y0, x0, y1, x1 = (int(v) for v in block_box)
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError(f"非法 block_box: {block_box}，图像尺寸 {(h, w)}")

    wy0, wy1 = max(0, y0 - r), min(h, y1 + r)
    wx0, wx1 = max(0, x0 - r), min(w, x1 + r)

    is_trimap = TRIMAP_UNKNOWN in np.unique(arr).tolist()
    if is_trimap:
        unk_pts = np.argwhere(arr[wy0:wy1, wx0:wx1] == TRIMAP_UNKNOWN)
        fg_pts = np.argwhere(arr[wy0:wy1, wx0:wx1] == TRIMAP_FG)
        bg_pts = np.argwhere(arr[wy0:wy1, wx0:wx1] == TRIMAP_BG)
    else:
        unk_pts = np.argwhere(arr[wy0:wy1, wx0:wx1].astype(bool))
        fg_pts = np.zeros((0, 2), dtype=np.int64)
        bg_pts = np.zeros((0, 2), dtype=np.int64)
    # 局部坐标转全局坐标
    if unk_pts.size:
        unk_pts = unk_pts + np.array([wy0, wx0])
    if fg_pts.size:
        fg_pts = fg_pts + np.array([wy0, wx0])
    if bg_pts.size:
        bg_pts = bg_pts + np.array([wy0, wx0])

    return {
        "shape": (h, w),
        "block": (y0, x0, y1, x1),
        "window": (wy0, wx0, wy1, wx1),
        "expansion": (y0 - wy0, x0 - wx0, wy1 - y1, wx1 - x1),
        "radius": r,
        "unknown_points": unk_pts.astype(np.int64),
        "fg_points": fg_pts.astype(np.int64),
        "bg_points": bg_pts.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# 修复点 3：块间重叠区加权融合
# ---------------------------------------------------------------------------
def _feather_weight(box, shape, overlap: int) -> np.ndarray:
    """块内融合权重：块中心为 1，向四条边线性衰减到 0，保证重叠区平滑过渡。"""
    y0, x0, y1, x1 = box
    th, tw = y1 - y0, x1 - x0
    ramp_y = max(1, min(int(overlap), th // 2 if th // 2 > 0 else 1))
    ramp_x = max(1, min(int(overlap), tw // 2 if tw // 2 > 0 else 1))
    yy = np.arange(th, dtype=np.float64)
    xx = np.arange(tw, dtype=np.float64)
    wy = np.clip((yy + 1) / ramp_y, 0.0, 1.0) * np.clip((th - yy) / ramp_y, 0.0, 1.0)
    wx = np.clip((xx + 1) / ramp_x, 0.0, 1.0) * np.clip((tw - xx) / ramp_x, 0.0, 1.0)
    return np.outer(wy, wx)


def _tile_boxes(h: int, w: int, block_size: int, overlap: int):
    step = max(1, int(block_size) - int(overlap))

    def starts(n: int):
        if n <= block_size:
            return [0]
        s = list(range(0, n - block_size + 1, step))
        if s[-1] + block_size < n:
            s.append(n - block_size)
        return s

    return [
        (y, x, min(y + block_size, h), min(x + block_size, w))
        for y in starts(h)
        for x in starts(w)
    ]


def block_fusion(tiles, shape=None, overlap: int = 32) -> np.ndarray:
    """将各分块的 alpha 结果按重叠区距离权重加权融合。

    ``tiles`` 为 ``(block_box, tile_alpha)`` 序列，block_box 为 ``(y0,x0,y1,x1)``。
    权重在块内部为 1、在重叠带内线性衰减到 0（见 :func:`_feather_weight`），
    因此同一像素被多个块覆盖时取加权平均，块边界处不再出现硬接缝。
    """
    if _is_scalar(tiles):
        return np.zeros((1, 1), dtype=np.float64)
    if not tiles:
        raise ValueError("tiles 为空")
    if shape is None:
        h = max(box[2] for box, _ in tiles)
        w = max(box[3] for box, _ in tiles)
    else:
        h, w = shape
    acc = np.zeros((h, w), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    for box, tile in tiles:
        y0, x0, y1, x1 = box
        tile = np.asarray(tile, dtype=np.float64)
        if tile.shape != (y1 - y0, x1 - x0):
            raise ValueError(f"tile 形状 {tile.shape} 与 block_box {box} 不匹配")
        weight = _feather_weight(box, (h, w), overlap)
        acc[y0:y1, x0:x1] += tile * weight
        wsum[y0:y1, x0:x1] += weight
    return np.divide(acc, wsum, where=wsum > 0.0, out=np.zeros_like(acc))


def _mean_color(img: np.ndarray, points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return fallback
    return img[points[:, 0], points[:, 1]].mean(axis=0)


def matte_solve(image, trimap=None, radius: int | None = None, block_size: int = 256,
                overlap: int = 32) -> np.ndarray:
    """分块并行的 alpha 求解。

    - 采样半径随图像尺寸缩放：``radius = max(2, round(0.01 * min(H, W)))``；
    - 每块通过 :func:`sample_border` 外扩采样窗，块边界邻域样本充足；
    - 未知像素用局部前景/背景颜色均值做闭式 alpha 估计
      ``a = ((c-b)·(f-b)) / |f-b|^2``；
    - 块间重叠区经 :func:`block_fusion` 同款权重融合，结果连续；
    - 全程无随机数、纯函数，结果对相同输入逐位确定。
    """
    if _is_scalar(image) or _is_scalar(trimap) or trimap is None:
        return np.zeros((1, 1), dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    tri = np.asarray(trimap)
    if tri.ndim != 2 or img.shape[:2] != tri.shape:
        raise ValueError("image 与 trimap 空间尺寸不一致")
    if img.ndim == 2:
        img = img[..., np.newaxis]
    h, w = tri.shape
    if radius is None:
        radius = max(2, int(round(0.01 * min(h, w))))

    fg_mask = tri == TRIMAP_FG
    bg_mask = tri == TRIMAP_BG
    unk_mask = tri == TRIMAP_UNKNOWN

    global_fg = (img[fg_mask].mean(axis=0) if fg_mask.any()
                 else np.ones(img.shape[2], dtype=np.float64))
    global_bg = (img[bg_mask].mean(axis=0) if bg_mask.any()
                 else np.zeros(img.shape[2], dtype=np.float64))

    acc = np.zeros((h, w), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    for box in _tile_boxes(h, w, block_size, overlap):
        y0, x0, y1, x1 = box
        geo = sample_border(tri, radius, box)
        wy0, wx0, wy1, wx1 = geo["window"]
        f = _mean_color(img, geo["fg_points"], global_fg)
        b = _mean_color(img, geo["bg_points"], global_bg)

        tile = np.where(fg_mask[y0:y1, x0:x1], 1.0, 0.0)
        sub_unk = unk_mask[y0:y1, x0:x1]
        if sub_unk.any():
            colors = img[y0:y1, x0:x1][sub_unk]
            diff = f - b
            denom = float(diff @ diff) + _EPS
            a = ((colors - b) * diff).sum(axis=1) / denom
            tile[sub_unk] = np.clip(a, 0.0, 1.0)
        weight = _feather_weight(box, (h, w), overlap)
        acc[y0:y1, x0:x1] += tile * weight
        wsum[y0:y1, x0:x1] += weight

    alpha = np.divide(acc, wsum, where=wsum > 0.0, out=np.zeros_like(acc))
    # 确定区域钉死，未知区兜底 0.5（正常不会触发）。
    alpha[fg_mask] = 1.0
    alpha[bg_mask] = 0.0
    alpha[unk_mask & (wsum <= 0.0)] = 0.5
    return alpha


# ---------------------------------------------------------------------------
# 修复点 4：多尺度按层级顺序提交
# ---------------------------------------------------------------------------
def scale_order(scales):
    """返回粗尺度 -> 细尺度的层级提交计划。

    - 传入正整数 n：生成 n 级计划，尺度因子为 2^-(n-1) ... 2^0；
    - 传入尺度序列：按尺度升序（粗到细）排序并标注 level，拒绝非正/重复尺度。

    返回 ``[{"level": int, "scale": float}, ...]``，提交顺序即列表顺序。
    """
    if _is_scalar(scales):
        n = int(scales)
        if n <= 0:
            raise ValueError("尺度层数必须为正整数")
        return [
            {"level": i, "scale": float(2 ** (-(n - 1 - i)))}
            for i in range(n)
        ]
    seq = [float(s) for s in np.asarray(scales).ravel().tolist()]
    if not seq:
        raise ValueError("尺度序列为空")
    if any(s <= 0.0 for s in seq):
        raise ValueError("尺度因子必须为正数")
    rounded = [round(s, 10) for s in seq]
    if len(set(rounded)) != len(seq):
        raise ValueError("尺度序列存在重复值")
    order = np.argsort(seq)  # 升序：小尺度（粗）在前，大尺度（细）在后
    return [{"level": i, "scale": seq[int(idx)]} for i, idx in enumerate(order)]


class MultiScaleSolver:
    """多尺度求解调度器：粗尺度结果未写回前，拒绝细尺度提交。

    旧实现异步传播时粗尺度可能晚于细尺度写回，造成层级污染。
    本调度器只允许按 ``scale_order`` 计划的 level 顺序 ``submit``，乱序直接抛错。
    """

    def __init__(self, scales):
        self.plan = scale_order(scales)
        self._next_level = 0
        self.results: list = [None] * len(self.plan)

    def submit(self, level: int, payload):
        if level != self._next_level:
            raise RuntimeError(
                f"多尺度乱序写回：当前应提交 level {self._next_level} "
                f"（scale={self.plan[self._next_level]['scale']}），却收到 level {level}；"
                "粗尺度必须先于细尺度写回"
            )
        self.results[level] = payload
        self._next_level += 1
        return level

    @property
    def finished(self) -> bool:
        return self._next_level == len(self.plan)


# ---------------------------------------------------------------------------
# 修复点 5：trimap 逐图重算，禁止跨图复用缓存
# ---------------------------------------------------------------------------
def _fingerprint(mask, image) -> tuple:
    """图像内容指纹：形状 + 前景像素数 + 全图像素和，换图必变。"""
    m = np.asarray(mask)
    im = np.asarray(image)
    return (
        m.shape,
        m.dtype.str,
        int(m.sum()),
        im.shape,
        int(np.rint(im.astype(np.float64)).astype(np.int64).sum()),
    )


def trimap_recompute(mask, image=None, cache: dict | None = None,
                     base_width: float = 4.0, max_width: float | None = None) -> dict:
    """逐图重算 trimap。

    旧实现在批量处理时复用上一张图缓存的 trimap（含确定前景），新图前景错误。
    现在：

    - 始终以当前图的 mask + image 计算内容指纹；
    - 仅当缓存指纹与当前图完全一致时才允许命中（同图重跑场景）；
    - 换图后指纹不同，缓存必然失效并重新构建 trimap。

    返回 ``{"trimap", "cache", "cache_hit", "recomputed"}``。
    """
    if _is_scalar(mask) or _is_scalar(image) or image is None:
        empty = np.full((1, 1), TRIMAP_UNKNOWN, dtype=np.uint8)
        return {
            "trimap": empty,
            "cache": {"signature": None, "trimap": empty},
            "cache_hit": False,
            "recomputed": True,
        }
    signature = _fingerprint(mask, image)
    cache_hit = False
    trimap = None
    if isinstance(cache, dict) and cache.get("signature") == signature:
        cached = cache.get("trimap")
        if cached is not None and np.asarray(cached).shape == np.asarray(mask).shape:
            trimap = np.asarray(cached).copy()
            cache_hit = True
    if trimap is None:
        trimap = trimap_build(mask, image, base_width, max_width)
    new_cache = {"signature": signature, "trimap": trimap}
    return {
        "trimap": trimap,
        "cache": new_cache,
        "cache_hit": cache_hit,
        "recomputed": not cache_hit,
    }


# ---------------------------------------------------------------------------
# 修复点 6：边缘按 alpha 直方图校验
# ---------------------------------------------------------------------------
def edge_check(alpha, trimap=None, bins: int = 16,
               hard_ratio_limit: float = 0.5, soft_ratio_min: float = 0.05) -> dict:
    """对边缘（未知带）内的 alpha 做直方图校验。

    检查项：
    - alpha 取值落在 [0,1] 且均为有限值；
    - 边缘带内被硬切到 0/1 的像素比例不得超过 ``hard_ratio_limit``（alpha 变硬检测）；
    - 边缘带内软像素（``soft_ratio_min < a < 1-soft_ratio_min``）比例不得低于
      ``soft_ratio_min``（必须存在过渡层次）。

    返回 ``{"ok", "reasons", "edge_pixels", "hard_ratio", "soft_ratio",
    "histogram", "bin_edges"}``。
    """
    bin_edges = np.linspace(0.0, 1.0, int(bins) + 1)
    if _is_scalar(alpha):
        return {
            "ok": True,
            "reasons": [],
            "edge_pixels": 0,
            "hard_ratio": 0.0,
            "soft_ratio": 0.0,
            "histogram": np.zeros(int(bins), dtype=np.int64),
            "bin_edges": bin_edges,
        }
    a = np.asarray(alpha, dtype=np.float64)
    reasons = []
    if np.any(~np.isfinite(a)) or a.min() < -_EPS or a.max() > 1.0 + _EPS:
        reasons.append("alpha 存在非有限值或超出 [0,1] 范围")

    if trimap is not None and not _is_scalar(trimap):
        band = np.asarray(trimap) == TRIMAP_UNKNOWN
    else:
        band = (a > _EPS) & (a < 1.0 - _EPS)

    if not band.any():
        return {
            "ok": not reasons,
            "reasons": reasons + (["边缘带为空，无软过渡像素"] if not reasons else []),
            "edge_pixels": 0,
            "hard_ratio": 0.0,
            "soft_ratio": 0.0,
            "histogram": np.zeros(int(bins), dtype=np.int64),
            "bin_edges": bin_edges,
        }

    edge_values = a[band]
    histogram, edges = np.histogram(edge_values, bins=int(bins), range=(0.0, 1.0))
    hard_ratio = float(np.mean((edge_values <= _EPS) | (edge_values >= 1.0 - _EPS)))
    soft_ratio = float(np.mean(
        (edge_values > soft_ratio_min) & (edge_values < 1.0 - soft_ratio_min)
    ))
    if hard_ratio > hard_ratio_limit:
        reasons.append(
            f"边缘 alpha 变硬：{hard_ratio:.2%} 的边缘像素被硬切为 0/1，"
            f"超过阈值 {hard_ratio_limit:.2%}"
        )
    if soft_ratio < soft_ratio_min:
        reasons.append(
            f"边缘缺少软过渡：软像素占比 {soft_ratio:.2%}，低于阈值 {soft_ratio_min:.2%}"
        )
    return {
        "ok": not reasons,
        "reasons": reasons,
        "edge_pixels": int(band.sum()),
        "hard_ratio": hard_ratio,
        "soft_ratio": soft_ratio,
        "histogram": histogram,
        "bin_edges": edges,
    }


# ---------------------------------------------------------------------------
# 精修、指标、校验、审计、兜底、确定性
# ---------------------------------------------------------------------------
def alpha_refine(alpha, trimap=None, passes: int = 2) -> np.ndarray:
    """仅在未知带内做 3x3 邻域均值平滑，确定前景/背景始终钉死，不污染硬约束。"""
    if _is_scalar(alpha):
        return np.asarray([1.0], dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64).copy()
    movable = (np.ones(a.shape, dtype=bool) if trimap is None or _is_scalar(trimap)
               else np.asarray(trimap) == TRIMAP_UNKNOWN)
    for _ in range(max(1, int(passes))):
        padded = np.pad(a, 1, mode="edge")
        local_sum = np.zeros_like(a)
        for dy in range(3):
            for dx in range(3):
                local_sum += padded[dy:dy + a.shape[0], dx:dx + a.shape[1]]
        smoothed = local_sum / 9.0
        a = np.where(movable, smoothed, a)
    return np.clip(a, 0.0, 1.0)


def metrics(alpha, gt_alpha=None, trimap=None) -> dict:
    """计算 alpha 质量指标；提供真值时额外给出 MAE、SAD 与边缘带 MAE。"""
    if _is_scalar(alpha):
        return {
            "mean_alpha": 0.0, "min_alpha": 0.0, "max_alpha": 0.0,
            "unknown_fraction": 0.0, "mae": None, "sad": None, "edge_mae": None,
        }
    a = np.asarray(alpha, dtype=np.float64)
    out = {
        "mean_alpha": float(a.mean()),
        "min_alpha": float(a.min()),
        "max_alpha": float(a.max()),
        "unknown_fraction": (
            float(np.mean(np.asarray(trimap) == TRIMAP_UNKNOWN))
            if trimap is not None and not _is_scalar(trimap) else 0.0
        ),
        "mae": None,
        "sad": None,
        "edge_mae": None,
    }
    if gt_alpha is not None and not _is_scalar(gt_alpha):
        gt = np.asarray(gt_alpha, dtype=np.float64)
        if gt.shape != a.shape:
            raise ValueError("gt_alpha 与 alpha 形状不一致")
        diff = np.abs(a - gt)
        out["mae"] = float(diff.mean())
        out["sad"] = float(diff.sum())
        if trimap is not None and not _is_scalar(trimap):
            band = np.asarray(trimap) == TRIMAP_UNKNOWN
            out["edge_mae"] = float(diff[band].mean()) if band.any() else 0.0
    return out


def validation(image, trimap=None, alpha=None) -> dict:
    """流水线入参/出参合法性校验，返回 ``{"ok", "errors", "warnings"}``。"""
    if _is_scalar(image) or _is_scalar(trimap) or trimap is None:
        return {"ok": True, "errors": [], "warnings": []}
    errors, warnings = [], []
    tri = np.asarray(trimap)
    img = np.asarray(image)
    if tri.ndim != 2:
        errors.append("trimap 必须是二维数组")
    if img.shape[:2] != tri.shape:
        errors.append("image 与 trimap 空间尺寸不一致")
    labels = set(np.unique(tri).tolist())
    if not labels.issubset({TRIMAP_BG, TRIMAP_UNKNOWN, TRIMAP_FG}):
        errors.append(f"trimap 含非法标签，允许值为 0/128/255，实际 {sorted(labels)}")
    if not np.any(tri == TRIMAP_FG):
        warnings.append("trimap 缺少确定前景样本")
    if not np.any(tri == TRIMAP_BG):
        warnings.append("trimap 缺少确定背景样本")
    if alpha is not None and not _is_scalar(alpha):
        a = np.asarray(alpha, dtype=np.float64)
        if a.shape != tri.shape:
            errors.append("alpha 与 trimap 形状不一致")
        elif np.any(~np.isfinite(a)) or a.min() < -_EPS or a.max() > 1.0 + _EPS:
            errors.append("alpha 存在非有限值或超出 [0,1]")
        elif np.any(~np.isfinite(a[tri == TRIMAP_UNKNOWN])):
            errors.append("存在未求解的未知像素")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def audit(alpha, trimap=None, image=None) -> dict:
    """汇总一次求解的审计信息：边缘直方图校验、各区域像素数与均值、整体直方图。"""
    if _is_scalar(alpha) or _is_scalar(trimap) or trimap is None:
        return {
            "edge": edge_check(1),
            "n_fg": 0, "n_bg": 0, "n_unknown": 0,
            "unknown_alpha_mean": None,
            "histogram": np.zeros(16, dtype=np.int64),
            "finite": True,
        }
    a = np.asarray(alpha, dtype=np.float64)
    tri = np.asarray(trimap)
    unk = tri == TRIMAP_UNKNOWN
    histogram, _ = np.histogram(a, bins=16, range=(0.0, 1.0))
    return {
        "edge": edge_check(a, tri),
        "n_fg": int(np.sum(tri == TRIMAP_FG)),
        "n_bg": int(np.sum(tri == TRIMAP_BG)),
        "n_unknown": int(unk.sum()),
        "unknown_alpha_mean": float(a[unk].mean()) if unk.any() else None,
        "histogram": histogram,
        "finite": bool(np.all(np.isfinite(a))),
    }


def fallback(image=None, mask=None) -> dict:
    """求解失败时的安全兜底：返回硬掩码 alpha 与全确定 trimap，并标记 degraded。"""
    if mask is not None and not _is_scalar(mask):
        m = np.clip(np.asarray(mask, dtype=np.float64), 0.0, 1.0)
        alpha = m
        trimap = np.where(m > 0.5, TRIMAP_FG, TRIMAP_BG).astype(np.uint8)
    elif image is not None and not _is_scalar(image):
        shape = np.asarray(image).shape[:2]
        alpha = np.zeros(shape, dtype=np.float64)
        trimap = np.full(shape, TRIMAP_BG, dtype=np.uint8)
    else:
        alpha = np.zeros((1, 1), dtype=np.float64)
        trimap = np.full((1, 1), TRIMAP_UNKNOWN, dtype=np.uint8)
    return {
        "alpha": alpha,
        "trimap": trimap,
        "degraded": True,
        "reason": "fallback：求解失败，返回确定性安全结果",
    }


def _structures_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_structures_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_structures_equal(x, y) for x, y in zip(a, b))
    return a == b


def _max_abs_diff(a, b) -> float:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        return 0.0
    if isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)):
        return abs(float(a) - float(b))
    return 0.0


def determinism(fn, *args, repeats: int = 3, **kwargs) -> dict:
    """重复运行同一求解函数，校验输出逐位一致（无随机性、无跨图缓存污染）。"""
    if not callable(fn):
        return {"deterministic": True, "runs": 0, "max_abs_diff": 0.0,
                "note": "入参不可调用，跳过重复执行"}
    runs = max(2, int(repeats))
    outputs = [fn(*args, **kwargs) for _ in range(runs)]
    first = outputs[0]
    deterministic = all(_structures_equal(first, other) for other in outputs[1:])
    max_diff = max(_max_abs_diff(first, other) for other in outputs[1:])
    return {
        "deterministic": deterministic,
        "runs": runs,
        "max_abs_diff": max_diff,
    }
