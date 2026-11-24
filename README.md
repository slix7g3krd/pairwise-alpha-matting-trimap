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
