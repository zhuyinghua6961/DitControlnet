# 实验问题记录（SDXL Teacher 生成）

本文记录在 SDXL + IP-Adapter teacher 生成过程中出现的问题、定位过程与最终解决办法，便于复现与排查。

## 1. 模型缓存与下载

问题
- Hugging Face 镜像下载时进程中断，缓存目录不一致，导致“下载了但本地为空”。

现象
- 控制台显示下载体量很大，但指定缓存目录为空。

处理
- 统一使用固定缓存目录。
- 下载与生成分离，生成阶段使用 `--offline`。

结论
- 下载阶段只写 `/mnt/fast18/models_cache`，生成阶段始终 `--offline` + `--model-cache /mnt/fast18/models_cache`。

## 2. VAE fp16 解码 NaN

问题
- 输出全黑或严重花屏，日志出现 `invalid value encountered in cast`。

原因
- SDXL 在 fp16 解码下出现 NaN。

处理
- 生成时启用 VAE upcast（`--upcast-vae`），并在该模式下禁用 autocast。

结论
- `--upcast-vae` 必须开启，否则黑图率显著升高。

## 3. IP-Adapter 注入过强

问题
- 结构严重破坏、重影、点阵网纹，明显不是正常风格迁移。

原因
- `ip_adapter_scale` 过大（1.0），采样不稳定。

处理
- 下调 `ip_adapter_scale` 到 0.30，并允许失败时降到 0.25。

结论
- `ip_adapter_scale` 取 0.25–0.35 更稳。

## 4. Euler 调度器不稳定

问题
- EulerDiscreteScheduler 在 SDXL img2img + IP-Adapter 下坏图率偏高。

原因
- Euler 路径在该组合下更容易导致采样轨迹不稳定。

处理
- teacher 生成改用 DDIM。

结论
- 生成阶段默认使用 DDIM，质量更稳。

## 8. 结构一致性筛选

问题
- 少量样本出现“结构崩坏/碎片化”，但不会被饱和度或低方差筛选拦住。

处理
- 加入边缘结构相似度筛选（输出与内容图的边缘相似度过低则判坏）。
- 失败样本重试时优先降低 `strength`，其次降低 `ip_adapter_scale`，同时降低 `cfg`。

结论
- 需要“内容一致性”筛选才能稳定降低坏图比例。

## 9. 内容一致性过滤（CLIP 可选）

问题
- 部分样本结构可读但内容漂移明显，像素统计无法拦住。

处理
- 使用 CLIP 图像特征计算输出与内容图相似度，低于阈值的样本进入重试或丢弃。
- 阈值建议由正常样本分布的 10% 分位数确定。

结论
- CLIP 内容相似度是最有效的“可选增强筛选”。

## 10. 高频伪影过滤

问题
- 刷痕/半色调/网纹类伪影不会被饱和度、低方差或边缘相似度完全拦住。

处理
- 使用 Sobel 能量比（输出/内容）作为高频异常指标。
- 若 `hf_ratio` 过大（例如 > 2.5），判为坏图并重试。

结论
- 高频能量比能有效过滤“纹理刷痕过重”的坏样本。

## 11. A/B/C 质量分级

问题
- 仅靠“通过/失败”会丢失可用但偏脏的样本。

处理
- 依据 CLIP 相似度与双尺度边缘相似度划分 A/B/C：
  - A：`content_sim >= 0.34` 且 `edge_sim64/128 >= 0.25`
  - B：其余通过者（低权重）
  - C：`content_sim < 0.30` 或 `edge_sim64/128 < 0.20` 或高频异常

结论
- A 样本直接入库，B 样本低权重使用，C 样本重试或丢弃。

## 5. 低显存模式冲突

问题
- `--low-vram` 触发 `meta` device 错误。

原因
- IP-Adapter 与 sequential CPU offload 组合不稳定。

处理
- 关闭 `--low-vram`。

结论
- 3090 单卡下不使用 `--low-vram`。

## 6. RAM 管理

问题
- 长时间生成可能导致 RAM 缓慢上涨。

处理
- 每张图后显式关闭 PIL 图像对象。
- 定期 `gc.collect()`。
- 不做批量缓存，流式写盘。

结论
- `--gc-interval` 建议 10 或 20。

## 7. 质量控制

问题
- 少量样本仍会出现极端坏图。

处理
- 增加自动筛选。
- 失败样本重试时降低 `ip_adapter_scale`。

结论
- 开启 `--quality-filter` 并设置 `--max-retries 2`。
