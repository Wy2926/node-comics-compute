# 流水线并发

普通图片在独立单线程池中冻结结果，本地 `persist` / `persist_queue` 记录落盘耗时与排队；完成事务后才释放页面缓冲区、进入上传。中心仍收到 v3 原有阶段和耗时字段。

调度循环读取两个接单收据的已提交快照，避免读取接单状态时等待大结果事务；回滚不发布快照。

文档按工作区共享状态锁，磁盘总预算使用短锁预留、结算。下载和 ZIP 预留已知大小；解析子进程在 PNG／临时解压文件增长前申请额度，按 1MiB 合并请求。失败或取消后计入残留文件，归还未使用额度；终态授权后清理工作区。

在 `services/classic-engine` 的测试环境运行：

```sh
python -m pytest tests/test_analysis_ocr.py tests/test_colors.py tests/test_node_scheduling.py tests/test_node_transport.py tests/test_document_workspace.py tests/test_document_node.py tests/test_pipeline_pressure.py ../../packages/document-files/tests/test_files.py
```

Linux 沙箱验收：`NODE_TEST_SANDBOX=1 python -m unittest discover -s services/compute-node/linux -p test_parser_sandbox.py`（仓库根目录，已安装共享文件包）。

并发配比须在目标机器用相同输入、模型和资产比较吞吐、CPU 时间、RSS、输出摘要与取消恢复；修改 `local_pages` 时同时评估 `engine.threads`、`inpaint_slots` 和渲染线程预算。
