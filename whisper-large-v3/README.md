---
license: apache-2.0
language:
  - zh
pipeline_tag: audio-text-to-text
tags:
  - 部署指南
hardware: NPU
---

# 引言
Whisper 是一种先进的自动语音识别（ASR）和语音翻译模型，在 OpenAI 的 Alec Radford 等人的论文 通过大规模弱监督实现鲁棒语音识别中提出。经过超过500万小时标注数据的训练，Whisper 表现出了在零样本设置下泛化到多种数据集和领域的能力。
Whisper large-v3 在架构上与之前的 large 和 large-v2 模型相同，除了以下微小差异：

    *频谱图输入使用了128个梅尔频率bins;
    *添加了一个新的粤语语言标记;
    
Whisper large-v3 模型是在100万小时的弱标签音频和使用Whisper large-v2 收集的400万小时伪标签音频上进行训练的。该模型在这个混合数据集上训练了2.0个轮次。

large-v3 模型在多种语言上表现出了改进的性能，相比Whisper large-v2，错误率降低了10%至20%。

# 一、运行环境准备
### 表 1 版本配套表
|配套|版本 |环境准备指导 |
|--|--|--|
|Python|3.10.12|-|
|torch|2.5.1+cpu|-|
|torch_npu|2.5.1|-|

 **硬件设备**
|设备型号|NPU配置|
|---|---|
|Atlas 800I A2 910B|	1卡| 

可以通过下面的命令安装torch2.5.1+cpu版本
```
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.5.1 -i https://mirrors.huaweicloud.com/repository/pypi/simple --no-cache-dir
```
参考torch相关版本：
```
[root:whisper-large-v3]$ pip list | grep torch
pytorch-lightning                        2.6.0
pytorch-metric-learning                  2.5.0
torch                                    2.5.1+cpu
torch-audiomentations                    0.12.0
torch-npu                                2.5.1
torch_pitch_shift                        1.2.5
torchaudio                               2.5.1+cpu
torchcodec                               0.8.1
torchdata                                0.11.0
torchmetrics                             1.8.2
torchvision                              0.20.1+cpu
```

``` 
# 补充安装依赖包
pip install transformers==4.57.3 -i https://mirrors.aliyun.com/pypi/simple
pip install datasets==4.4.1 -i https://mirrors.aliyun.com/pypi/simple
pip install accelerate==1.12.0 -i https://mirrors.aliyun.com/pypi/simple
pip install soundfile -i https://mirrors.aliyun.com/pypi/simple

pip install librosa -i https://mirrors.aliyun.com/pypi/simple

pip install modelscope -i https://mirrors.aliyun.com/pypi/simple

pip install decorator==5.2.1 -i https://mirrors.aliyun.com/pypi/simple
```

# 二、下载模型权重
## 1安装modelscope
```
pip install modelscope
```

## 2	下载模型权重文件
```
mkdir whisper-large-v3
modelscope download --model AI-ModelScope/whisper-large-v3  --local_dir ./whisper-large-v3
```

# 三、安装依赖
## 1安装ffmpeg
ffmpeg在语音处理时也会被使用；
```
apt update && apt install ffmpeg
```
## 2安装其他软件
根据实际中出现的问题，安装其他软件
```
pip install transformers datasets[audio] accelerate
[root:whisper-large-v3]$ pip list | grep transformers
transformers                             4.57.3
[root:whisper-large-v3]$ pip list | grep datasets
datasets                                 4.4.1
[root:whisper-large-v3]$ pip list | grep accelerate
accelerate                               1.12.0
```

# 四、推理部署
## 1创建测试数据

```
mkdir dataset
cp -r ../../dataset/* ./dataset
cd dataset
```
比如：
```
[root:dataset]$ ll
total 15872
-rw-r--r-- 1 root root   76770 11月 29 11:28 input.wav
-rw-r--r-- 1 root root 6497604 11月 29 11:28 speech1.wav
-rw-r--r-- 1 root root 2567488 11月 29 11:28 speech2.wav
```

## 2编写测试脚本
支持单个文件、多个文件、整个目录下的文件的输入进行测试。
比如： infer_demo.sh为：
```
#单个文件输入：
python infer.py --model ./ --input ./dataset/speech1.wav --output ./output/test1.csv
#多个文件输入：
python infer.py --model ./ --input ./dataset/speech1.wav ./dataset/speech2.wav --output ./output/test2.csv
#整个目录输入：
python infer.py --model ./ --input ./dataset --output ./output/test3.csv
```

推理脚本infer.py参考如下：
```
import os
import sys
import argparse
import csv
import time
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 核心配置
SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.oga', '.w64', '.aiff', '.aif', '.au', '.raw', '.mp3', '.m4a', '.3gp'}
NPU_DEVICE = "npu:0"
DEVICE = None
MAX_TARGET_POSITIONS = 448
SAFE_MAX_NEW_TOKENS = 438
TARGET_SAMPLING_RATE = 16000
KEEP_SINGLE_TXT = True  # 是否保留单个文本文件
CONCURRENT_THREADS = 4  # 并发线程数
BATCH_SIZE = 2  # 批量推理大小
PREVIEW_THRESHOLD = 4  # 预览阈值从8改为4
# 全局变量
GLOBAL_OUTPUT_DIR = None
GLOBAL_CSV_FILENAME = None
GLOBAL_PROCESSOR = None
GLOBAL_MODEL = None
GLOBAL_SHOW_DETAIL = False

def check_npu_environment() -> bool:
    """检测昇腾NPU环境"""
    try:
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            return False
        global DEVICE
        DEVICE = NPU_DEVICE
        print(f"🔧 推理设备：昇腾NPU（{torch.npu.device_count()}卡）| 并发线程：{CONCURRENT_THREADS} | 批量大小：{BATCH_SIZE}")
        return True
    except Exception:
        return False

def init_device() -> None:
    """初始化设备"""
    global DEVICE
    if not check_npu_environment():
        if torch.cuda.is_available():
            DEVICE = "cuda:0"
            print(f"🔧 推理设备：NVIDIA GPU（{torch.cuda.device_count()}卡）| 并发线程：{CONCURRENT_THREADS} | 批量大小：{BATCH_SIZE}")
        else:
            DEVICE = "cpu"
            print(f"🔧 推理设备：CPU | 并发线程：{CONCURRENT_THREADS} | 批量大小：{BATCH_SIZE}")

def init_model(model_dir: str) -> None:
    """初始化Whisper模型和处理器"""
    global GLOBAL_PROCESSOR, GLOBAL_MODEL
    model_dir = Path(model_dir)
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(f"模型文件缺失：{missing_files}")

    print(f"🤖 加载模型：本地Whisper模型（{model_dir.absolute()}）")
    GLOBAL_PROCESSOR = WhisperProcessor.from_pretrained(
        str(model_dir), local_files_only=True, cache_dir=str(model_dir)
    )
    GLOBAL_MODEL = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(DEVICE).eval()

    if hasattr(GLOBAL_MODEL.generation_config, "forced_decoder_ids"):
        GLOBAL_MODEL.generation_config.forced_decoder_ids = None

def parse_output_path(output_param: str) -> tuple[Path, str]:
    """解析输出路径"""
    output_path = Path(output_param)
    if output_path.suffix.lower() == ".csv":
        csv_filename = output_path.name
        output_dir = output_path.parent
    else:
        csv_filename = "transcript_results.csv"
        output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.absolute(), csv_filename

def get_audio_duration(audio_path: Path, framerate: int = None, audio_data: np.ndarray = None) -> float:
    """获取音频时长"""
    try:
        if audio_path.suffix.lower() in ('.wav', '.flac', '.ogg', '.oga', '.w64', '.aiff', '.aif', '.au', '.raw'):
            import soundfile as sf
            return round(sf.info(audio_path).duration, 2)
        elif framerate and audio_data is not None:
            return round(len(audio_data) / framerate, 2)
        return 0.0
    except Exception:
        return 0.0

def resample_audio(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """音频重采样"""
    if orig_sr == TARGET_SAMPLING_RATE:
        return audio
    if orig_sr % TARGET_SAMPLING_RATE == 0:
        return audio[::orig_sr // TARGET_SAMPLING_RATE]
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLING_RATE)
    except ImportError:
        raise ImportError("需安装librosa：pip install librosa --no-deps")
    except Exception as e:
        raise RuntimeError(f"重采样失败：{str(e)}")

def load_audio(audio_file: Path) -> tuple[np.ndarray, float, str]:
    """加载单个音频文件"""
    error_msg = ""
    audio_data = None
    duration = 0.0
    try:
        suffix = audio_file.suffix.lower()
        if suffix in ('.wav', '.flac', '.ogg', '.oga', '.w64', '.aiff', '.aif', '.au', '.raw'):
            import soundfile as sf
            audio_data, framerate = sf.read(audio_file)
            duration = get_audio_duration(audio_file, framerate, audio_data)
        elif suffix in ('.mp3', '.m4a', '.3gp'):
            from pydub import AudioSegment
            load_func = {'.mp3': AudioSegment.from_mp3, '.m4a': AudioSegment.from_file, '.3gp': AudioSegment.from_file}[suffix]
            audio_seg = load_func(str(audio_file))
            framerate = audio_seg.frame_rate
            audio_data = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
            audio_data = audio_data / (2 ** (8 * audio_seg.sample_width - 1)) if audio_seg.sample_width > 2 else audio_data / 32768.0
            duration = get_audio_duration(audio_file, framerate, audio_data)

        if audio_data is not None and audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = resample_audio(audio_data, framerate) if audio_data is not None else None
        audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32) if audio_data is not None else None
    except Exception as e:
        error_msg = str(e)
    return audio_data, duration, error_msg

def process_audio_batch(audio_batch: list[Path]) -> list[dict]:
    """处理音频批次"""
    batch_results = []
    batch_audio_data = []
    batch_audio_info = []

    # 批量加载音频
    for audio_file in audio_batch:
        start_time = time.time()
        audio_data, duration, error_msg = load_audio(audio_file)
        process_time = round(time.time() - start_time, 2)
        info = {
            "file_name": audio_file.name,
            "file_path": str(audio_file.absolute()),
            "audio_duration": duration,
            "process_time": process_time,
            "error": error_msg,
            "transcript": "",
            "status": "failed"
        }
        batch_audio_info.append(info)
        batch_audio_data.append(audio_data if error_msg == "" and audio_data is not None else None)

    # 批量推理
    valid_indices = [i for i, d in enumerate(batch_audio_data) if d is not None]
    valid_audio = [batch_audio_data[i] for i in valid_indices]
    if valid_audio:
        try:
            inputs = GLOBAL_PROCESSOR(
                valid_audio, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt",
                padding=True, truncation=True, return_attention_mask=True
            ).to(DEVICE)
            with torch.no_grad():
                predicted_ids = GLOBAL_MODEL.generate(
                    **inputs,
                    language="chinese",
                    task="transcribe",
                    max_new_tokens=SAFE_MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
                    repetition_penalty=1.1,
                    max_length=MAX_TARGET_POSITIONS
                )
            transcripts = GLOBAL_PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)
            # 填充结果
            for idx, trans in zip(valid_indices, transcripts):
                trans = trans.strip()
                batch_audio_info[idx]["transcript"] = trans
                batch_audio_info[idx]["status"] = "success" if trans else "empty"
                if not trans:
                    batch_audio_info[idx]["error"] = "转写结果为空，音频可能无有效语音"
                # 保存单个文件
                if KEEP_SINGLE_TXT and trans and batch_audio_info[idx]["status"] == "success":
                    output_txt = GLOBAL_OUTPUT_DIR / f"{Path(batch_audio_info[idx]['file_name']).stem}_transcript.txt"
                    with open(output_txt, "w", encoding="utf-8") as f:
                        f.write(trans)
        except Exception as e:
            err_msg = f"推理失败：{str(e)}"
            for i in valid_indices:
                batch_audio_info[i]["error"] = err_msg
                batch_audio_info[i]["status"] = "failed"

    # 打印详情和收集结果
    for info in batch_audio_info:
        if info["error"] == "" and info["status"] not in ["success", "empty"]:
            info["status"] = "empty" if not info["transcript"] else "failed"

        if GLOBAL_SHOW_DETAIL:
            icon = "✅" if info["status"] == "success" else "❌" if info["status"] == "failed" else "⚠️"
            print(f"{icon} {info['file_name']} | 状态：{info['status']}")
            if info["status"] == "success" and info["transcript"]:
                preview = info["transcript"][:100] + "..." if len(info["transcript"]) > 100 else info["transcript"]
                print(f"   📝 转写预览：{preview}\n")

        batch_results.append(info)
    return batch_results

def collect_audio_files_fast(input_paths: list[str]) -> list[Path]:
    """快速收集音频文件（优化耗时）"""
    audio_files = []
    seen_paths = set()  # 用集合快速去重，比字典更高效
    suffix_set = SUPPORTED_FORMATS

    for path_str in input_paths:
        path = Path(path_str).absolute()
        if not path.exists():
            print(f"⚠️  输入路径不存在，跳过：{path_str}")
            continue

        # 处理文件：直接判断后缀，无需递归
        if path.is_file():
            if path.suffix.lower() in suffix_set:
                path_str = str(path)
                if path_str not in seen_paths:
                    seen_paths.add(path_str)
                    audio_files.append(path)
            continue

        # 处理目录：使用生成器表达式快速遍历，减少内存占用
        for file in path.rglob("*"):
            if file.is_file() and file.suffix.lower() in suffix_set:
                file_str = str(file)
                if file_str not in seen_paths:
                    seen_paths.add(file_str)
                    audio_files.append(file)

    return audio_files

def split_batches_fast(audio_files: list[Path], batch_size: int) -> list[list[Path]]:
    """快速切分批次（使用列表切片，最高效的方式）"""
    return [audio_files[i:i+batch_size] for i in range(0, len(audio_files), batch_size)]

def main():
    """主函数"""
    global GLOBAL_OUTPUT_DIR, GLOBAL_CSV_FILENAME, GLOBAL_SHOW_DETAIL
    parser = argparse.ArgumentParser(description="Whisper批量转写工具（昇腾NPU并发+极速优化）")
    parser.add_argument("--model", "-m", required=True, help="本地模型目录")
    parser.add_argument("--input", "-i", required=True, nargs='+', help="多文件/多目录输入")
    parser.add_argument("--output", "-o", required=True, help="输出路径（目录/CSV文件）")
    args = parser.parse_args()

    # 初始化输出路径
    GLOBAL_OUTPUT_DIR, GLOBAL_CSV_FILENAME = parse_output_path(args.output)

    # 初始化设备和模型
    init_device()
    init_model(args.model)

    # 快速收集音频文件（核心优化：减少耗时）
    audio_files = collect_audio_files_fast(args.input)
    if not audio_files:
        print(f"❌ 未找到任何支持的音频文件")
        return

    # 设置预览开关（阈值改为4）
    GLOBAL_SHOW_DETAIL = len(audio_files) <= PREVIEW_THRESHOLD
    file_count_tip = f"（显示单个文件状态和转写预览）" if GLOBAL_SHOW_DETAIL else f"（仅显示汇总结果）"
    print(f"\n📋 待处理音频总数：{len(audio_files)}个 {file_count_tip}")

    # 快速切分批次（核心优化：减少耗时）
    audio_batches = split_batches_fast(audio_files, BATCH_SIZE)
    print(f"📦 切分为批次总数：{len(audio_batches)}个（每批{BATCH_SIZE}个）\n")

    # 并发处理
    all_results = []
    start_total_time = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
        future_map = {executor.submit(process_audio_batch, batch): batch for batch in audio_batches}
        if not GLOBAL_SHOW_DETAIL:
            print(f"🚀 开始批量处理{len(audio_files)}个音频文件...\n")
        # 快速遍历未来对象
        for future in as_completed(future_map):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"❌ 批次处理失败：{str(e)}")

    # 计算耗时
    total_time = round(time.time() - start_total_time, 2)
    avg_time = round(total_time / len(audio_files), 2) if audio_files else 0

    # 保存CSV
    csv_file = GLOBAL_OUTPUT_DIR / GLOBAL_CSV_FILENAME
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "file_path", "audio_duration", "process_time", "transcript", "status", "error"])
        writer.writeheader()
        writer.writerows(all_results)

    # 统计结果
    success = sum(1 for r in all_results if r["status"] == "success")
    empty = sum(1 for r in all_results if r["status"] == "empty")
    failed = len(all_results) - success - empty

    # 打印报告
    print(f"\n===== 处理完成报告 =====")
    print(f"📊 结果已保存为CSV：{csv_file.absolute()}")
    print(f"📈 处理统计：成功{success}个 | 空结果{empty}个 | 失败{failed}个 | 总计{len(all_results)}个")
    print(f"⏱️  耗时统计：总耗时{total_time}秒 | 平均每个文件{avg_time}秒")

if __name__ == "__main__":
    main()
```

## 3推理结果
```
[root:whisper-large-v3]$ bash infer_demo.sh
🔧 推理设备：昇腾NPU（1卡）| 并发线程：4 | 批量大小：2
🤖 加载模型：本地Whisper模型（.../whisper-large-v3）
📋 待处理音频总数：7个 （仅显示汇总结果）
📦 切分为批次总数：4个（每批2个）
🚀 开始批量处理7个音频文件...
===== 处理完成报告 =====
📊 结果已保存为CSV：.../whisper-large-v3/output/test1.csv
📈 处理统计：成功7个 | 空结果0个 | 失败0个 | 总计7个
⏱️  耗时统计：总耗时6.18秒 | 平均每个文件0.88秒
```


