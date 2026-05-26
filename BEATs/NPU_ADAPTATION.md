# BEATs NPU 适配指导

## 1. 基准信息

- 上游：<https://github.com/microsoft/unilm.git>
- 子目录：`beats/`
- 分支：`master`
- 本地基准 commit：`833df7e7832e5064a281131ee64a481afa8e5b95`
- 本次复核日期：2026-05-25

> 远端最新 commit 需要在可联网环境执行 `git -C BEATs/upstream ls-remote origin master` 复核；当前适配仍基于上述本地 upstream。

## 2. 环境搭建

```bash
python3 -m venv BEATs/.venv
source BEATs/.venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio
# NPU 环境按 CANN 版本安装匹配 torch-npu
pip install torch-npu
```

`requirements.txt` 为历史整环境导出，不建议作为最小依赖。

## 3. 权重下载

官方权重在 <https://github.com/microsoft/unilm/tree/master/beats> 列出的 OneDrive 链接中。下载 fine-tuned checkpoint 到：

```text
BEATs/weights/model.pt
```

脚本：

```bash
./BEATs/scripts/download_weights.sh BEATs/weights
BEATS_WEIGHT_URL=<direct-url> ./BEATs/scripts/download_weights.sh BEATs/weights BEATs/weights/model.pt
```

离线环境可直接拷贝 `.pt`，推理时用 `--checkpoint` 指定。

## 4. 测试数据

```bash
./BEATs/scripts/download_test_data.sh BEATs/test_data
```

输出：`BEATs/test_data/dummy_1s_16k.wav` 和 `dummy_1s_16k.wav.meta.json`。该 dummy wav 只验证链路。脚本不会联网，已有文件会直接复用。

正式 ESC-50 评测数据按指定目录准备：

```bash
./BEATs/scripts/prepare_esc50_data.sh BEATs/eval_data/esc50
OFFLINE=1 ./BEATs/scripts/prepare_esc50_data.sh BEATs/eval_data/esc50
```

离线模式要求 `BEATs/eval_data/esc50/ESC-50-master/` 或 `ESC-50-master.zip` 已存在；输出固定 manifest 和 metadata，CPU/NPU 对比复用同一份 manifest。

## 5. 应用 patch

```bash
cd BEATs/upstream
git apply ../patches/0001-add-npu-fbank-device-support.patch
cp ../infer.py beats/infer.py
```

patch 内容：`beats/BEATs.py::preprocess()` 中 fbank CPU 回退，完成后把特征搬回输入设备。

## 6. infer.py 参数

- `--checkpoint`：BEATs fine-tuned checkpoint。
- `--wav`：16 kHz 或可被 torchaudio 重采样的 wav。
- `--device`：`npu`、`cpu` 或 `cuda`，默认 `npu`。
- `--warmup` / `--repeat`：性能测试循环。
- `--topk`：输出 top-k 标签。

## 7. CPU 推理

```bash
cd BEATs/upstream/beats
python infer.py --checkpoint ../../weights/model.pt --wav ../../test_data/dummy_1s_16k.wav --device cpu --warmup 0 --repeat 1
```

## 8. NPU 推理

```bash
cd BEATs/upstream/beats
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py --checkpoint ../../weights/model.pt --wav ../../test_data/dummy_1s_16k.wav --device npu --warmup 5 --repeat 20
```

## 9. 常见问题

- `ModuleNotFoundError: torch_npu`：CPU 验证不需要导入 torch_npu；确认使用的是新版 `infer.py`，且未传 `--device npu`。
- fbank 设备错误：确认 patch 已应用到 `BEATs/upstream/beats/BEATs.py`。
- 输出 label 为空：checkpoint 可能不是 fine-tuned 分类权重，或缺少 `label_dict`。

## 10. 完整验收

`dummy_1s_16k.wav` 只用于链路 smoke test。正式验收需固定官方 checkpoint，并按 `ACCEPTANCE_PLAN.md` 覆盖 representation / AudioSet 分类、ESC-50 或 AudioSet 精度、batch、RTF/RTFx、峰值 HBM 和稳定性。
