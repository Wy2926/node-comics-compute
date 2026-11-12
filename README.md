# Node Comics Compute

独立图像计算节点：OCR、抹字、嵌字、覆盖层输出，以及 PDF/ZIP 文件处理。通过 HTTPS 连接 `node-comics-backend`，无需后端或插件源码。

- [Windows 便携包](services/compute-node/README.md)：安装进度、双击升级
- [Linux NVIDIA](services/compute-node/linux/README.md)
- [Apple Silicon Mac](services/classic-engine/README.md#apple-silicon-mac)
- [引擎开发](services/classic-engine/README.md) · [配置与恢复](docs/NODE_CONFIGURATION.md)
- 文件解析包：`packages/document-files/`

## 检查

```sh
cd services/compute-node
go test ./...
go vet ./...
python -m pytest test_build_release.py test_portable.py test_run_node.py -q
```

Python 打包测试需要 pytest。引擎依赖与 GPU 验证见引擎 README；普通打包不下载模型。

源码起点：[node-comics@e0fb68b](https://github.com/Wy2926/node-comics/commit/e0fb68b1177ad0c5844fbe490bc3171753da40e2)。[GPL-3.0-only](LICENSE)，[第三方声明](services/classic-engine/THIRD_PARTY.md)。
