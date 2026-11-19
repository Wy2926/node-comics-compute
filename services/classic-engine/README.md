# 图像引擎

Python 3.12 + uv。在本目录按 GPU 选择一个环境：

| 后端 | 安装 | 后续命令前缀 |
| --- | --- | --- |
| NVIDIA CUDA 12.6 | `uv sync --locked --extra test` | `uv run` |
| Windows AMD | `uv sync --project environments/amd --locked --group test` | `uv run --project environments/amd` |
| Apple Silicon，macOS 14+ | `uv sync --project environments/macos --locked --group test` | `uv run --project environments/macos` |

复制 [node.example.json](node.example.json) 为私有 `node.local.json`，设置身份、模型路径和 `engine.backend`（`cuda`、`amd` 或 `mps`）。以下以 CUDA 为例：

```sh
uv run python -m tools.prepare_mtu
uv run python -m classic_node check --config node.local.json
uv run python -m classic_node run --config node.local.json
uv run python -m pytest -q
```

`prepare_mtu` 一次准备检测、Hayai OCR、抹字和字体；默认读取 `models/hayai`。三个后端都直接采样原图取色，无取色模型。已有单独校验过的 Hayai 目录可用 `engine.ocr_model` 指定。`check` 校验并预热，`run` 才接单。旧锁的资产须在独立目录重新准备，保留私有配置和恢复数据。

[流水线、实测与离线验证](../../docs/PIPELINE.md) · [配置与恢复](../../docs/NODE_CONFIGURATION.md) · [第三方来源与修改](THIRD_PARTY.md)
