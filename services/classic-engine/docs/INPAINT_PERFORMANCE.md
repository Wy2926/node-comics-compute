# 抹字耗时

普通图片的 `event=page_stage stage=inpaint`、文件页的 `event=document_page_stage` 在本地节点日志记录数值指标，耗时单位为秒。

| 字段 | 范围 |
| --- | --- |
| `inpaint_queue` | 等待页面计算槽 |
| `inpaint_prepare` | 原始掩膜准备、气泡内缩及连通块统计 |
| `inpaint_solid_fill` | 纯色气泡采样与填充 |
| `inpaint_lama` | 逐块裁剪、GPU 等待、模型调用和贴回 |
| `inpaint_lock_wait` | 等待共享 GPU 捕获保护锁 |
| `inpaint_model` | 模型输入准备、推理、回传与恢复尺寸 |
| `inpaint_blocks` / `inpaint_solid_regions` | LaMa 调用次数 / 纯色填充匹配区域数 |
| `inpaint_graph_blocks` / `inpaint_eager_blocks` | AMD 图重放 / 常规推理次数 |

嵌套耗时不能相加；并发页面耗时之和不是整批墙钟时间。明细保存在本地，中心仍使用原有协议字段。

AMD 在启动预热时为 LaMa 原生 FP32 生成器捕获八种正方形输入：64 至 512，步长 64。图缓存有固定上限，运行中不捕获新图；更大尺寸、不同布局或捕获失败使用原生推理。捕获失败会验证原生路径，真实模型或设备错误仍然报错。

重放与上游输入准备共用调用方的 GPU 流，输出复制到每次调用自己的张量，完成后才释放模型锁，避免并发页面覆盖共享结果。原生缩放、补边、修复掩膜、窗口顺序和贴回路径仍由 MTU 执行。CUDA/MPS 沿用原生推理。性能与像素一致性须在目标 GPU、固定分析结果和相同输入上实测；模拟测试只检查调度与回退。
