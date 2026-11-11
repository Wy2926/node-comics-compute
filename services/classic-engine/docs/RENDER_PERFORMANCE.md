# 排版嵌字验证

在引擎目录使用已准备的资产运行；只调用 CPU/Qt，不启动节点或调用中心、LLM、GPU 模型：

```sh
python -m tools.validate_render_output --models .assets/models --output render.json --workers 2 --threads 4 --pages 8 --rounds 3 --language ko
```

- `--case many` 检查多段文字，`--case tiles` 检查长图分块；默认还包括稀疏、透明和无文字样本。
- `--samples private/cases.json` 添加固定检查点。JSON 数组每项包含相对清单目录的 `original`、`cleaned`、`analysis` 文件路径；译文使用固定测试句子。
- 在各自源码快照及对应资产上生成报告，比较相同样本的阶段耗时、吞吐、RSS、像素／输出字节／元数据摘要。预热、校验解码不计入耗时；RSS 求和可能重复统计共享页。
- 修改上游补丁后须重新准备资产，启动会校验锁文件和源码摘要。

横排字形轮廓缓存属于每个渲染线程，按实际 Qt 字体值和字形编号复用，最多 2,048 项、65,536 个路径元素。超出预算只影响复用；字体注册表变化时清空。缓存不保存译文、页面图片或排版结果。

真实字体、回退、变形和缓存所有权回归：

```sh
MTU_TEST_ASSETS=/path/to/assets python -m pytest -q tests/test_render_glyph_cache.py
```

Windows PowerShell 使用 `$env:MTU_TEST_ASSETS='E:\path\to\assets'` 设置资产目录。离线 Qt 测试不代表真实翻译效果、GPU 或中心连接验收。
