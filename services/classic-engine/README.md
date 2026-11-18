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

`check` 校验资产并预热；`run` 才接单。使用 Hayai 时，运行 `python -m tools.prepare_hayai --source <下载目录> --output <模型目录>`，配置 `engine.ocr_backend=hayai`、`engine.ocr_model` 指向该目录。仅准备 Hayai 的公共资产时给 `prepare_mtu` 加 `--ocr-backend hayai`。更新上游补丁时，在独立目录按新锁重新准备资产。

[流水线、实测与离线验证](../../docs/PIPELINE.md) · [配置与恢复](../../docs/NODE_CONFIGURATION.md) · [第三方来源与修改](THIRD_PARTY.md)
