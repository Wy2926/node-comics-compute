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

AMD 使用 `uv sync --project environments/amd --locked --group test`。各环境独立，显式选择 `engine.backend`，不自动降级或混装。

## Apple Silicon Mac

原生 arm64、macOS 14 或更新版本。在本目录执行：

```sh
uv sync --project environments/macos --locked --group test
uv run --project environments/macos --locked python -m tools.prepare_mtu
```

私有 `node.local.json` 设置 `engine.backend: "mps"`、`engine.gpu: 0`，模型和字体路径沿用示例。检测、取色、气泡和 LaMa 使用 MPS；经典 OCR 的日文模型使用 MPS，PP-OCR 使用 CPU ONNX。MPS 计算串行保护，多页并发仍可重叠下载、嵌字和交付。

```sh
uv run --project environments/macos --locked python -m classic_node check --config node.local.json
uv run --project environments/macos --locked python -m classic_node run --config node.local.json
```

使用 Hayai 时，先运行 `tools.prepare_hayai --source <下载目录> --output <模型目录>`，再设置 `engine.ocr_backend: "hayai"`、`engine.ocr_model` 为该模型目录、`engine.ocr_batch_size: 8`。已有模型可继续使用；路径相对配置文件解析。仅准备 Hayai 公共资产时可给 `prepare_mtu` 加 `--ocr-backend hayai`。

升级遵循[恢复规则](docs/NODE_OPERATIONS.md)。上游补丁变化时，在独立目录按新锁文件重新准备资产，再停机更新源码和资产清单；保留私有配置与恢复数据。

Mac 实际模型回归（资产目录需包含经典 OCR 模型）：

```sh
MTU_TEST_BACKEND=mps MTU_TEST_ASSETS=/absolute/path/assets \
  uv run --project environments/macos --locked python -m pytest -q \
  tests/test_render_glyph_cache.py tests/test_mtu_integration.py \
  -k 'outline or native_rearrange or long_page_cross_seam'
```

## 验证

`tools.prepare_hayai` 准备可选 Hayai OCR；`tools.validate_mtu`、`tools.benchmark_pages`、`tools.benchmark_ocr` 用于离线效果和性能验证，参数见 `--help`。样本与输出由调用者指定。

[OCR 耗时与并发对比](docs/OCR_PERFORMANCE.md)：阶段耗时、模型锁等待、批量、吞吐与识别摘要。

[抹字耗时](docs/INPAINT_PERFORMANCE.md)：LaMa 调用、GPU 等待、AMD 图重放与预热约束。

[排版嵌字验证](docs/RENDER_PERFORMANCE.md)：离线 Qt、双进程耗时、结果摘要与字体轮廓缓存边界。

真实模型测试设置 `MTU_TEST_ASSETS`；默认测试不证明 GPU 或实图效果。中心联调在后端仓库显式设置 `COMICS_COMPUTE_ROOT`。

[引擎约束](ENGINE.md) · [配置](../../docs/NODE_CONFIGURATION.md) · [第三方来源](THIRD_PARTY.md)
