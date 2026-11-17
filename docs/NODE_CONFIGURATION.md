# 节点配置

以 [node.example.json](../services/classic-engine/node.example.json) 和 `classic_node/config.py` 为字段及默认值来源。

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

## 中心连接与调度

节点从 `control_url` 建立 WebSocket 控制通道，单图与文件共用一个容量上报入口；TLS、独立 CA 和代理策略复用 HTTPS 配置。代理须允许 WebSocket Upgrade，原图、结果与租约心跳走 HTTPS。

服务端管理后台预设 `performance_weight`（1–1000，默认 1）和 `execution_slots`；权重越大越优先，执行位为硬上限。节点只上报实际能力与本地可用资源。流程与消息定义由后端仓库 `docs/ARCHITECTURE.md`、`docs/COMPUTE_PROTOCOL.md` 维护。

## 文件任务

解析器在 `packages/document-files/`。默认 `document_require_sandbox=true`，要求 Linux Landlock ABI≥3 与 libseccomp；检查成功才声明 `document_protocol:2`。

Windows／macOS 可显式设置 `document_require_sandbox=false`，仍需通过解析自检；这是无沙箱模式，不自动回退。中心需开放文件领取并配置 document slots。

macOS 解析子进程保留 CPU 时间、输出文件大小、文件描述符和禁用 core dump 的限制；地址空间上限为启动时虚拟映射加 `parser_memory_bytes`，避免系统预留的巨大共享映射使默认 1 GiB 上限在启动时即失败。这限制新增虚拟地址空间，不是 RSS 硬限额；解析仍逐页处理并校验像素、页数、字节与磁盘预算。

`state_dir/documents/` 必须持久化并有磁盘容量限制；源文件、规范化页、检查点和 ZIP 共用预算。节点不配置 LLM 或 R2 密钥，只使用中心签发的精确临时授权。

[运行与恢复](../services/classic-engine/docs/NODE_OPERATIONS.md)
