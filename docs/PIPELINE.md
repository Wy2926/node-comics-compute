# 图像与文件流水线

节点通过 `compute.pull.v1` WebSocket 领取单图或文件工作，输入、租约心跳和结果走 HTTPS。节点负责计算，翻译由中心完成。

| 阶段 | 行为与边界 |
| --- | --- |
| 接单与输入 | 按租约、下载并发和页面内存预算接单。校验输入摘要、长度、格式、尺寸及规范化状态；文件先在独立工作区拆页，拆页与单图共用计算预算。 |
| 检测与识别 | DBNet 长图分块保留原图短边（受 `detect_size` 限制），重叠区概率图合并后统一提框。BallonsTranslator 分组；classic 按语言选模型，Hayai 按段识别，截断文本不授权抹字。 |
| 分析检查点 | 黑白文字跳过颜色模型；彩色段按代表行取色。气泡、CTD 字形细化和源笔画保护生成修复掩膜；尺寸、文字、几何和掩膜一起冻结。无字跳过抹字，计算错误不能作为无字提交。 |
| 翻译与抹字 | 提交分析后，中心翻译与本地抹字并行，等译文不占计算线程。纯色气泡先填充，剩余块由 FP32 LaMa 修复；取消在模型批次／块间生效。 |
| 排版与交付 | Qt 根据冻结译文、字体和气泡排版。空译文留白，缺少段 ID 或不支持的字体拒绝渲染。输出无损差分覆盖；长图先完整排版再分块。持久化完成后释放缓冲并上传；未知交付重发冻结结果。 |

主要入口：`classic_node/agent.py`、`pipeline.py`、`document_runner.py`（调度），`mtu_engine/engine.py`（图像阶段），`classic_node/protocol.py`（输出）。部署、排空与恢复见[节点配置](NODE_CONFIGURATION.md)。

## 并发与资源

- 下载、分析／抹字、渲染、落盘和交付分别有界。页面、原生掩膜和渲染进程传输缓冲共用 `resident_bytes`。可选参考掩膜每页最多 64 MiB；内存不足不增加并发。文件拆页及 ZIP 另受工作区磁盘预算约束。
- CUDA／AMD 以 `local_pages` 限制计算并发；MPS 一次派发一项分析／抹字，I/O 和渲染可重叠。`engine.threads`、渲染进程与页面数要一起评估；并发提高吞吐不等于降低单页延迟。
- AMD Hayai 使用原生 HIP Graph：可用显存至少 8 GiB 时最多保留 8 组状态，否则 4 组。仅捕获与其他 GPU 模型调用互斥；捕获失败退回原生 eager。AMD 取色复用位置系数，LaMa 在启动期准备固定形状的图。缓存不跨页复用识别结果。

## 调优依据

RX 6900 XT 16 GiB / Ryzen 7 5800X，`720×14400` 彩色 WebP，Hayai、FP32 检测、CPU 4 线程。按相同形状预热后各测 5 次。分析包括检查点编码，不含下载、翻译和排版。

| 本地指标 | 基线 `60bc66a` | 优化后，原批量 | 优化后，长图批量 |
| --- | ---: | ---: | ---: |
| 检测 / OCR / 取色批量 | 4 / 4 / 4 | 4 / 4 / 4 | 1 / 8 / 16 |
| 热分析中位数 | 9.12 s | 5.63 s | 4.34 s |
| 其中取色 / 掩膜细化 | 5.44 / 0.75 s | 2.55 / 0.31 s | 1.40 / 0.32 s |
| GPU 忙碌率 / 峰值已分配显存 | 38.8% / 2.69 GiB | 50.4% / 2.69 GiB | 60.3% / 1.84 GiB |

GPU 忙碌率取 Windows 进程最忙引擎。空 MIOpen 缓存、完成节点预热后的首张分析由 32.07 s 降到 13.58 s。保留 `data/cache/miopen` 可复用卷积选型。相同批量时完整分析及固定译文成品字节一致；长图批量保留 28 个区域的文字、行框、修复和气泡掩膜，5 个区域颜色最多差 2/255，原始检测掩膜有 3 个像素差 1/255。

这类长图建议 `engine.detect_batch_size=1`、`engine.ocr_batch_size=8`、`engine.color_batch_size=16`、`engine.threads=4`。增加至 8 或 16 线程未缩短单页延迟；1、2、3 路同图吞吐分别为 0.212、0.251、0.259 页/s。普通短页应另测：21 个混合输入的调度对照中，2 路计算、OCR 批量 4 更合适。批量和 FP16 都可能改变识别或像素，参数并非通用默认值。Apple Silicon 31 页实测融合 FP16 比 FP32 图像阶段少耗时约 12%，OCR 一致但像素不逐点等价；以上均不是线上翻译耗时。

## 诊断与复测

`state_dir/logs/node.log` 的 `event=page_stage` 和 `event=document_page_stage` 记录本地阶段秒数。先分清排队（`*_queue`）、等模型锁（`*_lock_wait`）和计算，再看以下字段。嵌套耗时不能相加，并发页耗时之和不等于墙钟时间。

| 排查项 | 主要字段 |
| --- | --- |
| 检测／OCR | `analyze_detect`、`detect_filter`、`analyze_group`、`analyze_route`、`analyze_ocr`；Hayai 细分 `ocr_preprocess`、`ocr_infer`、`ocr_decode`、`ocr_capture` |
| 分析后处理 | `analyze_colors`、`analyze_color_gate`、`color_model_lines`、`analyze_bubbles`、`analyze_bubble_mask`、`analyze_refine`、`analyze_checkpoint` |
| 抹字与排版 | `inpaint_solid_fill`、`inpaint_blocks`、`inpaint_lama`、`inpaint_model`；`render_layout`、`render_diff`、`render_encode`、`persist` |

在引擎环境中使用同一组输入、模型、配置和固定译文，报告放在输入目录以外。Windows 工作目录与资产同盘。报告默认只含摘要；`benchmark_ocr --include-text` 才保存识别全文。

```sh
python -m tools.benchmark_ocr --config node.local.json --input page.webp --concurrency 1 --repeats 5 --output ocr.json
python -m tools.benchmark_pages --backend amd --models <assets>/models --hayai-model <hayai> --input <pages> --output long.json --threads 4 --render-workers 1 --detect-batch-size 1 --ocr-batch-size 8 --color-batch-size 16 --cache-mib 64 --repeats 5
```

`benchmark_pages` 的 `first_pass` 与 `runs` 区分首次处理和热运行，同时报告文字、坐标、像素及输出摘要。`--pipeline` 可验证真实调度与本地模拟中心；`tools.validate_render_output` 单独验证 CPU／Qt 输出。模型和字体实机回归须指定 `MTU_TEST_ASSETS`，普通单测不能证明 GPU 性能或中心连通。
