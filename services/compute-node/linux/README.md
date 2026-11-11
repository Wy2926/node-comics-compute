# Linux NVIDIA

需要 NVIDIA 驱动、Docker 与 NVIDIA Container Toolkit。仓库根目录构建：

```sh
docker build -f services/compute-node/linux/Dockerfile -t node-comics-compute:linux-cuda .
```

复制本目录 `node.example.json` 为 `node.local.json`，填写独立节点身份与中心 HTTPS 地址；不要提交凭据。

```sh
docker compose -f services/compute-node/linux/compose.yaml up -d
docker compose -f services/compute-node/linux/compose.yaml logs --tail 100 node
```

`node-state` 卷保存未交付结果与文件工作目录。更新前排空；保留卷，不以删除恢复数据处理故障。

## Vast 在线安装

将本计算仓库发布后，设置 HTTPS GitHub `origin`，对完整提交 SHA 生成模板：

```sh
python services/compute-node/linux/prepare_template.py --revision <完整SHA> --output <新模板.json>
```

生成器核对远端 bootstrap 字节并绑定摘要，将源仓库写入 `NODE_SOURCE_REPOSITORY`。填写模板中独立的节点 ID、令牌和资源 ID；平台环境保留原入口，依赖与模型按锁下载。仅有本地仓库时使用本地镜像构建。

在线安装显示 8 个阶段；模型、字体下载显示已下载量与百分比，总大小未知时只显示下载量。镜像构建复用同一资产下载进度。

文件解析默认要求 Landlock ABI≥3 与 libseccomp，探测失败只提供普通单图能力。配置与恢复见[节点配置](../../../docs/NODE_CONFIGURATION.md)。
