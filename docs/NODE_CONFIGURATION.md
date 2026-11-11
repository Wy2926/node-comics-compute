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

## 文件任务

解析器在 `packages/document-files/`。默认 `document_require_sandbox=true`，要求 Linux Landlock ABI≥3 与 libseccomp；检查成功才声明 `document_protocol:2`。

Windows 可显式设置 `document_require_sandbox=false`，仍需通过解析自检；这是无沙箱模式，不自动回退。中心需开放文件领取并配置 document slots。

`state_dir/documents/` 必须持久化并有磁盘容量限制；源文件、规范化页、检查点和 ZIP 共用预算。节点不配置 LLM 或 R2 密钥，只使用中心签发的精确临时授权。

[运行与恢复](../services/classic-engine/docs/NODE_OPERATIONS.md)
