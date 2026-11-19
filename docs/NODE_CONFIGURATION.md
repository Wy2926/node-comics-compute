# 节点配置

引擎目录的 `node.example.json` 和 `classic_node/config.py` 为字段及默认值来源。

| 字段 | 作用 |
| --- | --- |
| `control_url` | 中心 HTTPS origin |
| `node_id`、`node_token`、`resource_id` | 独立节点身份、凭据与物理资源标识 |
| `state_dir` | 私有持久恢复目录 |
| `local_pages`、`render_workers` | 本地计算并发 |
| `max_leases` | 包含下载、等文本与交付的租约总量 |
| `resident_bytes` | 页工作内存预算 |
| `engine` | GPU 后端、模型、字体、线程与 OCR 配置 |
| `proxy_url` | 节点通信代理，不修改系统代理 |

Windows 便携包使用 `data/node.json`；Linux 挂载 `node.local.json`。模型和字体从固定清单准备，运行时不下载。

OCR 仅使用 Hayai，`engine.ocr_batch_size` 为 1–8。升级时先排空，配套更新运行环境和资产；原来显式设置 `engine.ocr_backend=classic` 的节点需删除该字段再 `check`。旧示例中的 `ocr_min_batch_size`、`ocr_arena_*` 已无效，可删除。没有指定 OCR 后端的旧配置会直接使用随包的 `models/hayai`。

CUDA、AMD 和 MPS 均从原图采样取色。旧 `engine.color_batch_size` 会被忽略，可直接沿用私有配置；恢复中的页继续使用检查点保存的颜色。

`engine.detect_precision` 仅控制 DBNet。`fp16` 先在 FP32 权重上折叠卷积与 BatchNorm，再转 FP16 推理，输出概率图仍为 FP32。`auto` 在 MPS 选择这条路径，在 AMD/CUDA 使用 FP32。精度和批量可能改变 OCR、颜色或掩膜，选择依据见[流水线实测](PIPELINE.md#调优依据)。对照 CPU 双边滤波时，可在进程启动前设置 `OPENCV_OPENCL_RUNTIME=disabled`。

## 中心连接与调度

节点从 `control_url` 建立 `compute.pull.v1` WebSocket 控制通道，单图与文件共用一个领取入口；TLS、独立 CA 和显式代理复用 HTTPS 配置。代理须允许 WebSocket Upgrade，原图、结果与租约心跳走 HTTPS。与中心配套升级，保留状态目录。

服务端管理后台预设 `performance_weight`（1–1000，默认 1）和 `execution_slots`；权重越大越优先，执行位为硬上限。节点只上报实际能力与本地可用资源。流程与消息定义由后端仓库 `docs/ARCHITECTURE.md`、`docs/COMPUTE_PROTOCOL.md` 维护。

## 文件任务

解析器在 `packages/document-files/`。默认 `document_require_sandbox=true`，要求 Linux Landlock ABI≥3 与 libseccomp；检查成功才声明 `document_protocol:2`。

Windows／macOS 可显式设置 `document_require_sandbox=false`，仍需通过解析自检；这是无沙箱模式，不自动回退。接文件任务要求中心配置 `document_slots > 0`。

macOS 解析子进程保留 CPU 时间、输出文件大小、文件描述符和禁用 core dump 的限制；地址空间上限为启动时虚拟映射加 `parser_memory_bytes`，避免系统预留的巨大共享映射使默认 1 GiB 上限在启动时即失败。这限制新增虚拟地址空间，不是 RSS 硬限额；解析仍逐页处理并校验像素、页数、字节与磁盘预算。

`state_dir/documents/` 必须持久化并有磁盘容量限制；源文件、规范化页、检查点和 ZIP 共用预算。节点不配置 LLM 或 R2 密钥，只使用中心签发的精确临时授权。

## 运行与恢复

- 用中心连接和新鲜心跳判断节点在线。停止先停领，再排空；同一节点身份不得在两台机器同时运行。
- 重启通过持久化的领取编号和文件 `sync` 恢复。结果先写入 SQLite，断网后重传冻结结果，不重复请求翻译。只有确认终态后才清理文件工作区。
- 升级前处理未完成任务与待确认交付，保留 `state_dir`、私有配置及额外模型；勿删除数据库或工作区强行切换。模型／上游锁变化时须配套准备新资产。
- 运维日志位于 `state_dir/logs/node.log`，只记录阶段、错误码及数值指标。排查阶段耗时见[诊断与复测](PIPELINE.md#诊断与复测)。
