# 引擎约束

MTU 检测、分组与 OCR → 中心文本翻译／本地抹字 → Qt 嵌字 → 无损覆盖结果。上游固定版本与补丁见 `mtu_engine/upstream.lock.json`、`tools/prepare_mtu.py` 和 [THIRD_PARTY.md](THIRD_PARTY.md)。

- `classic_node/` 管理租约、传输、恢复和资源；`mtu_engine/` 管理图像算法；`packages/document-files` 管理受限文件解析。
- 输入先校验摘要、长度、格式、尺寸和规范化状态；损坏或资源超限在文本调用前失败。
- OCR 支持 classic 与 Hayai。后端为 cuda、amd 或 mps，显式选择；设备不兼容直接失败。
- 字体只接受固定 ID，按目标语言验证；冻结任务不因环境变化静默换字体。
- 空译文保留抹字留白；缺少 segment ID 拒绝。检测或绘制失败不能伪装成无字。
- 差分覆盖包含背景、文字和抗锯齿；二值 alpha，单帧无损 WebP。大图按协商格式分块，先完整排版再切块。
- 租约、下载、分析、渲染和交付分别有界；等待文本不占计算线程。取消与截止贯穿各阶段。
- 缓存淘汰不改变结果；恢复复用冻结分析与译文，不重新请求 LLM。

协议由后端仓库 `docs/COMPUTE_PROTOCOL.md` 与 OpenAPI 维护；节点实现入口是 `classic_node/protocol.py` 和 `transport.py`。
