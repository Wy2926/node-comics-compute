# Windows 节点

Windows 10/11 x64，支持 NVIDIA CUDA 或单张 RX 6900 XT 的 AMD 环境。需匹配显卡驱动，预留至少 25 GiB。

1. 解压 `NodeComicsNode.zip` 到固定可写目录，运行 `Start.cmd`。
2. 首次选 `cuda` 或 `amd`，下载并校验依赖、模型和字体，再填中心 HTTPS 地址与独立节点身份。
3. `Status.cmd` 检查 `connected=true` 和新鲜心跳；`Stop.cmd` 停止整个节点进程树。

`Enable-Autostart.cmd` / `Disable-Autostart.cmd` 管理当前用户登录启动项。关闭窗口不等于停止节点。

## 更新与数据

先停止，再覆盖新版程序，保留 `data/`、`payload/`、`.setup/`。纯代码升级复用环境与模型；后端或依赖不兼容时重新准备。换 GPU 后端使用新目录。

`data/node.json` 含明文节点凭据；`data/` 保存恢复状态和日志。未确认交付不能删除；升级前排空不兼容任务。代理与文件模式见[节点配置](../../docs/NODE_CONFIGURATION.md)。

## 构建

仓库根目录，PowerShell 5.1 和系统 tar：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File services/compute-node/build.ps1 -Version 0.3.0
```

构建下载锁定工具链并运行 Go、打包测试，输出 `artifacts/dist/NodeComicsNode.zip`；ZIP 不含大型运行环境。输出目录必须新建。

`tests/verify_clean_checkout.py` 检查干净源码重建；`tests/verify_release.py` 验证已准备发布包的真实 GPU、断网与恢复，参数见 `--help`。
