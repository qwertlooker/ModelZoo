# BEATs NPU 验证记录

验证日期：2026-05-25。当前机器无可用 NPU；NPU 命令保留为待目标环境补验。

## 1. upstream / commit

```bash
git -C BEATs/upstream rev-parse HEAD
# 833df7e7832e5064a281131ee64a481afa8e5b95
```

远端最新 commit 需在可联网环境执行：

```bash
git -C BEATs/upstream ls-remote origin master
```

## 2. patch 检查

```bash
git -C BEATs/upstream apply --check ../patches/0001-add-npu-fbank-device-support.patch
# 结果：通过
```

## 3. Python 语法检查

```bash
python3 -m py_compile BEATs/infer.py
# 结果：通过
```

## 4. 权重与测试数据

- 权重路径：`BEATs/weights/model.pt`，当前环境未下载，需用户从官方 OneDrive 或直链下载。
- 测试数据：`BEATs/test_data/dummy_1s_16k.wav`。

准备命令：

```bash
./BEATs/scripts/download_weights.sh BEATs/weights
./BEATs/scripts/download_test_data.sh BEATs/test_data
```

测试数据脚本已执行成功。

## 5. 当前环境 CPU 验证

当前系统 `python3` 缺少 `torch`，因此使用已有 `Canary-1B/.venv-cpu/bin/python` 做依赖可达性检查：

```bash
PYTHONPATH=BEATs/upstream/beats Canary-1B/.venv-cpu/bin/python BEATs/infer.py \
  --checkpoint BEATs/weights/model.pt \
  --wav BEATs/test_data/dummy_1s_16k.wav \
  --device cpu --warmup 0 --repeat 1
```

结果：脚本进入 CPU 分支并打印 `Using device: cpu`，随后因未下载权重阻塞：

```text
FileNotFoundError: BEATs/weights/model.pt
```

结论：当前环境 CPU 完整推理未完成；阻塞原因是 BEATs 官方 checkpoint 未下载。下载权重后按同一命令补验。

## 6. NPU 验证命令

```bash
cd BEATs/upstream
# 若尚未应用 patch：git apply ../patches/0001-add-npu-fbank-device-support.patch
cp ../infer.py beats/infer.py
cd beats
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint ../../weights/model.pt \
  --wav ../../test_data/dummy_1s_16k.wav \
  --device npu --warmup 5 --repeat 20
```

预期：输出 `Elapsed`、top-k label/prob，且无 fbank NPU Tensor 设备不匹配错误。

## 7. 已知限制

- dummy wav 不用于准确率评估。
- fbank 前处理在 CPU 执行，会有数据搬运开销。
