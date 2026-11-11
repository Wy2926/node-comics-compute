# 图像引擎

Python 3.12 + uv，默认 CUDA 12.6。在本目录执行：

```sh
uv sync --locked --extra test
uv run python -m tools.prepare_mtu
uv run python -m classic_node check --config node.local.json
uv run python -m classic_node run --config node.local.json
uv run pytest -q
```

先复制 `node.example.json` 为私有 `node.local.json`。`check` 验证资产与预热，`run` 才接单；运行时不下载资产。

AMD 使用 `uv sync --project environments/amd --locked --group test`；Apple Silicon 使用 `environments/macos`。各环境独立，显式选择 `engine.backend`，不自动降级或混装。

`tools.prepare_hayai` 准备可选 Hayai OCR；`tools.validate_mtu`、`tools.benchmark_pages`、`tools.benchmark_ocr` 用于离线效果和性能验证，参数见 `--help`。样本与输出由调用者指定。

[OCR 耗时与并发对比](docs/OCR_PERFORMANCE.md)：阶段耗时、模型锁等待、批量、吞吐与识别摘要。

真实模型测试设置 `MTU_TEST_ASSETS`；默认测试不证明 GPU 或实图效果。中心联调在后端仓库显式设置 `COMICS_COMPUTE_ROOT`。

[引擎约束](ENGINE.md) · [配置](../../docs/NODE_CONFIGURATION.md) · [第三方来源](THIRD_PARTY.md)
