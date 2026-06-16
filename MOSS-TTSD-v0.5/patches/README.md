# MOSS-TTSD-v0.5 patches

当前适配不修改 `MOSS-TTSD-v0.5/upstream/` 中的 GitHub 上游已有文件，因此没有 `.patch` 文件。

本目录下的 `infer.py`、`download_weights.py`、`prepare_test_data.py` 和验证文档均为当前 ModelZoo 适配新增文件，不进入上游 patch。

如果后续确需修改上游已有文件，按项目标准在 `MOSS-TTSD-v0.5/upstream/` 中完成修改并生成 patch，例如：

```bash
git -C MOSS-TTSD-v0.5/upstream diff -- <upstream_existing_file> > MOSS-TTSD-v0.5/patches/0001-xxx.patch
git -C MOSS-TTSD-v0.5/upstream apply --check ../patches/0001-xxx.patch
```
