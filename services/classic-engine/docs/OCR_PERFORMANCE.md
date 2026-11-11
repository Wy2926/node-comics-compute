# OCR 耗时

普通图片的 `event=page_stage stage=analyze`、文件页的 `event=document_page_stage` 将数值耗时写入 `state_dir/logs/node.log`，单位为秒。

| 字段 | 范围 |
| --- | --- |
| `analyze_queue` | 等待页面计算槽 |
| `analyze_detect` / `detect_lock_wait` | 检测总耗时 / 检测锁等待 |
| `analyze_group` / `analyze_route` | 分段 / classic 语言探测 |
| `analyze_ocr` | 正式识别，含锁等待 |
| `ocr_crop` / `ocr_preprocess` | Hayai 裁剪 / CPU 预处理 |
| `ocr_transfer` / `ocr_infer` / `ocr_decode` | Hayai 输入传输 / 原生推理及结果回传 / 文本解码 |
| `ocr_model_lock_wait` | Hayai 模型锁等待 |
| `ocr_lock_wait` | 旧协议汇总，包含 OCR 与颜色模型锁等待 |
| `analyze_colors` / `analyze_bubbles` | 颜色 / 气泡检测 |
| `analyze_bubble_mask` / `analyze_refine` / `analyze_checkpoint` | 气泡掩膜 / 抹字掩膜细化 / 检查点编码 |

`ocr_batches`、`ocr_paragraphs`、`ocr_output_tokens`、`ocr_truncated` 是计数；`ocr_batch_max` 是实际最大批量。嵌套耗时不能相加；并发页耗时之和不等于整批墙钟时间。新增明细仅用于本地诊断，中心使用原有字段。

AMD Hayai 使用上游 HIP Graph 解码，缓存最多四组状态，捕获失败按上游逻辑回退 eager；`ocr_graph_batches` / `ocr_eager_batches` 记录成功批次。`ocr_capture_wait` / `ocr_capture` / `ocr_graph_captures` 分别记录捕获等待、耗时和次数，包含在 `ocr_infer` 中。捕获与其他 GPU 模型调用互斥，避免 HIP 的跨线程同步冲突。Graph 的静态填充可能改变识别标点或文字，不能视为逐字等价。CUDA/MPS 仍使用 eager。

## 离线对比

在引擎目录、对应 GPU 环境中，使用完整的本地引擎配置及私有样本：

```sh
python -m tools.benchmark_ocr --config node.local.json --input page1.jpg page2.jpg --concurrency 1 2 4 --repeats 3 --output ocr-baseline.json
python -m tools.benchmark_ocr --config node.local.json --input page1.jpg page2.jpg --concurrency 1 2 4 --batch-size 8 --repeats 3 --output ocr-batch8.json
```

输出每页明细、阶段均值/P50/P95、整批吞吐、CPU 时间/平均占用核数、RSS 和 CUDA/HIP 的 PyTorch 显存峰值。计时前预热各页形状；内存受 `resident_bytes` 限制。Windows 工作目录需与模型同盘，避免上游气泡检测器的跨盘相对路径限制。

默认只保存识别摘要；`--include-text` 才保存全文与区域。输出使用新文件，保留前次对比。测试期间保持节点空闲；输出差异会保留报告并报错，模拟测试不代表 GPU 性能或识别质量。
