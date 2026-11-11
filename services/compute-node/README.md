# Windows 节点

Windows 10/11 x64，支持 NVIDIA CUDA 或单张 RX 6900 XT 的 AMD 环境。需匹配显卡驱动，预留至少 25 GiB。

1. 解压 `NodeComicsNode.zip` 到固定可写目录，运行 `Start.cmd`。
2. 首次选 `cuda` 或 `amd`，按 6 个阶段下载、安装并校验依赖、模型和字体，显示下载量与百分比，再填中心 HTTPS 地址与独立节点身份。总大小未知时显示已下载量；启动期间持续显示预热与连接等待状态。
3. `Status.cmd` 检查 `connected=true` 和新鲜心跳；`Stop.cmd` 停止整个节点进程树。

`Enable-Autostart.cmd` / `Disable-Autostart.cmd` 管理当前用户登录启动项。关闭窗口不等于停止节点。

## 更新与数据

双击 `Update.cmd`：检查本仓库最新稳定 Release，下载并校验新版，自动停止、切换版本；原先运行的节点会重新启动并等待心跳。已是最新版时直接提示，无需输入命令或手动覆盖文件。

纯代码升级复用环境与模型；依赖或资产变化时，利用 `.setup/` 缓存预先准备新环境再切换，需要额外磁盘空间。升级保留 `data/`、节点身份和恢复数据。未完成任务或未确认交付会阻止切换，原先运行的节点恢复运行后可再点升级；换 GPU 后端使用新目录。

切换失败自动恢复旧版；切换中断后再次双击 `Update.cmd` 恢复，再点 `Start.cmd`。`.setup/update/previous/` 的旧版备份保留至下次升级检查。共用程序目录的其他 `--home` 实例也须停止。

`data/node.json` 含明文节点凭据；`data/` 保存恢复状态和日志。未确认交付不能删除；升级前排空不兼容任务。代理与文件模式见[节点配置](../../docs/NODE_CONFIGURATION.md)。

## 构建

仓库根目录，PowerShell 5.1 和系统 tar：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File services/compute-node/build.ps1 -Version 0.3.0
```

构建下载锁定工具链并运行 Go、打包测试，输出 `artifacts/dist/NodeComicsNode.zip` 和 `.zip.sha256`；ZIP 不含大型运行环境。输出目录必须新建。

在线升级读取 `Wy2926/node-comics-compute` 的最新稳定 GitHub Release。发布标签为构建版本（如 `v0.3.0`），同时上传上述 ZIP 与 SHA-256 文件。旧包首次取得 `Update.cmd` 仍需停止后覆盖一次新版小包，保留 `data/`、`payload/`、`.setup/`。

`tests/verify_clean_checkout.py` 检查干净源码重建；`tests/verify_release.py` 验证已准备发布包的真实 GPU、断网与恢复，参数见 `--help`。
