# Apple Silicon Mac 节点

原生 arm64、macOS 14 或更新版本，预留至少 25 GiB，无需预装 Python 或 Homebrew。

1. 解压 `NodeComicsNode-macos-arm64.zip` 到固定可写目录，双击 `Start.command`。
2. 首次自动下载独立 Python、锁定依赖、模型及字体，然后打开 `data/node.json`。填写中心 HTTPS 地址、独立节点 ID、令牌和资源 ID，保存后再次启动。
3. `Status.command` 查看中心连接与新鲜心跳；`Stop.command` 排空并停止节点，关闭终端窗口不会停止后台服务。

`Configure.command` 编辑配置，`Check.command` 在节点停止后验证模型及 MPS。`Enable-Autostart.command`／`Disable-Autostart.command` 管理当前用户登录自启；每个 macOS 用户管理一个节点。

## 更新与恢复

`Update.command` 下载并校验本仓库最新稳定 Release 的 Mac ZIP。依赖／资产提前准备，排空后切换并恢复原运行状态；失败回滚，中断后再次启动或更新会先恢复。

本地 ZIP 更新：`./node.sh update /absolute/path/NodeComicsNode-macos-arm64.zip`，同目录需有 `.zip.sha256`。

更新保留 `data/` 和额外模型；未确认交付须处理完成后再切换。上一版保存在 `.setup/update/previous/`，下次更新时替换。移动目录前停止节点，移动后重新启动以更新服务路径。

首次准备包含 Hayai OCR。旧安装器发起的首次跨版本升级可能保留闲置依赖及权重，不会加载它们；不要清理回滚目录或私有模型。原配置若显式指定 `ocr_backend=classic`，先删除该字段。

未签名脚本可能需要右键“打开”或在“系统设置 → 隐私与安全性”允许打开。

## 构建

仓库根目录，Python 3.12 和 uv：

```sh
python3.12 services/compute-node/macos/build.py --version 0.3.1 --output services/compute-node/artifacts/macos/NodeComicsNode-macos-arm64
python3.12 -m pytest services/compute-node/macos/test_macos.py services/compute-node/test_build_release.py services/compute-node/test_run_node.py -q
```

输出目录必须新建。分发生成的 ZIP，不复制含身份和恢复数据的运行目录。发布时上传 ZIP 与 `.zip.sha256`，构建版本须与稳定 Release 标签一致。
