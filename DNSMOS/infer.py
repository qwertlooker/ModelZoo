import argparse
import concurrent.futures
import glob
import os
import random
import subprocess
from functools import lru_cache

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# 配置参数
SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01  # 秒
FRAME_SIZE = 320
HOP_LENGTH = 160
N_MELS = 120
DEFAULT_BATCH_SIZE = 4  # 默认批量大小

class NPUBoostScorer:
    def __init__(self, primary_model, p808_model, batch_size=DEFAULT_BATCH_SIZE):
        self.providers = ['CANNExecutionProvider']
        self.batch_size = batch_size
        self._verify_npu_environment()
        
        self.primary_sess = ort.InferenceSession(
            primary_model, 
            providers=self.providers,
            sess_options=self._get_optimized_session_options()
        )
        self.p808_sess = ort.InferenceSession(
            p808_model, 
            providers=self.providers,
            sess_options=self._get_optimized_session_options()
        )
        
        self._verify_model_outputs()
        print(f"[NPU加速] 初始化完成 | 批量大小: {self.batch_size} | 推理设备: CANNExecutionProvider")

    def _get_optimized_session_options(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = False
        return options

    def _verify_npu_environment(self):
        try:
            if 'CANNExecutionProvider' not in ort.get_all_providers():
                raise RuntimeError("未找到CANNExecutionProvider，请安装onnxruntime-cann")
            
            result = subprocess.run(
                ["npu-smi info"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"NPU设备异常: {result.stderr.strip()}")
            
            subprocess.run(
                ["npu-smi set -t performance -i 0"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("[NPU加速] 已切换至性能模式")
            
        except Exception as e:
            print(f"[错误] NPU环境验证失败: {str(e)}")
            exit(1)

    def _verify_model_outputs(self):
        p808_output_shape = self.p808_sess.get_outputs()[0].shape
        if len(p808_output_shape) != 2:
            print(f"[警告] P808模型输出维度为{len(p808_output_shape)}，预期为2维")
        else:
            print(f"[验证] P808模型输出形状: {p808_output_shape}（正常）")

    @lru_cache(maxsize=128)
    def _preprocess_audio(self, file_path):
        audio_data, input_sr = sf.read(file_path)
        original_duration = len(audio_data) / input_sr  # 计算原始音频时长（秒）
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(np.float32)
        else:
            audio_data = audio_data.astype(np.float32)
        
        if input_sr != SAMPLING_RATE:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=input_sr,
                target_sr=SAMPLING_RATE,
                res_type='fft'
            ).astype(np.float32)
        
        min_length = int(INPUT_LENGTH * SAMPLING_RATE)
        if len(audio_data) < min_length:
            repeat = (min_length // len(audio_data)) + 1
            audio_data = np.tile(audio_data, repeat)[:min_length]
        
        return audio_data, original_duration

    def _compute_melspec_batch(self, audio_batch):
        batch_mel = []
        for audio in audio_batch:
            min_len = HOP_LENGTH * 2 + FRAME_SIZE
            if len(audio) < min_len:
                audio = np.pad(audio, (0, min_len - len(audio)), mode='constant')
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=SAMPLING_RATE,
                n_fft=FRAME_SIZE + 1,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
            batch_mel.append(mel_spec.T)
        
        return np.stack(batch_mel, axis=0).astype(np.float32)

    def _poly_correction_batch(self, raw_scores, is_personalized):
        sig_raw, bak_raw, ovr_raw = raw_scores[:, 0], raw_scores[:, 1], raw_scores[:, 2]
        
        if is_personalized:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        
        return np.column_stack([
            p_sig(sig_raw),
            p_bak(bak_raw),
            p_ovr(ovr_raw)
        ])

    def process_batch(self, file_batch, is_personalized):
        batch_results = []
        audio_batch = []
        durations = []  # 存储每个音频的原始时长（秒）
        
        # 批量预处理
        for file_path in file_batch:
            audio, duration = self._preprocess_audio(file_path)
            audio_batch.append(audio)
            durations.append(duration)  # 记录原始音频时长
        
        # 生成批量输入
        batch_primary_input = []
        batch_p808_input = []
        for audio in audio_batch:
            hop_samples = SAMPLING_RATE
            total_samples = len(audio)
            num_windows = max(1, int(np.floor(total_samples / SAMPLING_RATE) - INPUT_LENGTH) + 1)
            mid_idx = num_windows // 2
            start = mid_idx * hop_samples
            end = start + int(INPUT_LENGTH * SAMPLING_RATE)
            window = audio[start:end]
            
            batch_primary_input.append(window[np.newaxis, :])
            batch_p808_input.append(window[:-160])
        
        # 批量推理
        batch_p808_mel = self._compute_melspec_batch(batch_p808_input)
        primary_input = np.concatenate(batch_primary_input, axis=0)
        raw_scores = self.primary_sess.run(None, {'input_1': primary_input})[0]
        
        # 适配P808模型2维输出
        p808_output = self.p808_sess.run(None, {'input_1': batch_p808_mel})[0]
        p808_scores = p808_output[:, 0]
        
        corrected_scores = self._poly_correction_batch(raw_scores, is_personalized)
        
        # 整理结果（保留len_in_sec）
        for i, file_path in enumerate(file_batch):
            batch_results.append({
                'filename': os.path.basename(file_path),
                'len_in_sec': round(durations[i], 4),  # 保留音频原始时长（秒）
                'MOS_SIG': round(corrected_scores[i, 0], 3),
                'MOS_BAK': round(corrected_scores[i, 1], 3),
                'MOS_OVRL': round(corrected_scores[i, 2], 3),
                'P808_MOS': round(p808_scores[i], 3)
            })
            # 控制台显示完整信息
            print(f"[批量处理] {batch_results[-1]['filename']} | 时长: {batch_results[-1]['len_in_sec']}s | SIG: {batch_results[-1]['MOS_SIG']} | BAK: {batch_results[-1]['MOS_BAK']} | OVRL: {batch_results[-1]['MOS_OVRL']} | P808: {batch_results[-1]['P808_MOS']} | 完成")
        
        return batch_results

def find_audio_files(root_dir):
    return glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)

def main():
    parser = argparse.ArgumentParser(description="NPU加速版DNSMOS评分工具（保留len_in_sec）")
    parser.add_argument('-t', '--test_dir', required=True, help='音频文件目录')
    parser.add_argument('-o', '--output_csv', required=True, help='结果输出CSV路径')
    parser.add_argument('-p', '--personalized', action='store_true', help='使用个性化MOS模型')
    parser.add_argument('--model_root', default='.', help='模型文件根目录')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='NPU批量大小（默认4）')
    args = parser.parse_args()

    # 模型路径配置
    p808_model_path = os.path.join(args.model_root, 'DNSMOS', 'model_v8.onnx')
    primary_model_path = os.path.join(
        args.model_root, 'pDNSMOS' if args.personalized else 'DNSMOS', 'sig_bak_ovr.onnx'
    )

    # 检查模型文件
    missing_models = [m for m in [primary_model_path, p808_model_path] if not os.path.exists(m)]
    if missing_models:
        print("错误：以下模型文件不存在：")
        for m in missing_models:
            print(f"  - {m}")
        return

    # 初始化NPU评分器
    scorer = NPUBoostScorer(primary_model_path, p808_model_path, args.batch_size)

    # 查找音频文件
    audio_files = find_audio_files(args.test_dir)
    if not audio_files:
        print(f"错误：在 {args.test_dir} 未找到WAV文件")
        return
    
    # 去重并限制最大数量
    audio_files = list(set(audio_files))
    if len(audio_files) > 250:
        audio_files = random.sample(audio_files, 250)
    
    # 按批量大小分组
    batches = [
        audio_files[i:i + args.batch_size] 
        for i in range(0, len(audio_files), args.batch_size)
    ]
    print(f"[NPU加速] 总文件数: {len(audio_files)} | 批次数: {len(batches)} | 批量大小: {args.batch_size}")

    # 并行处理批次
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(scorer.process_batch, batch, args.personalized)
            for batch in batches
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(batches), desc="NPU处理进度"):
            results.extend(future.result())

    # 保存结果（包含len_in_sec）
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    # 强制指定列顺序（filename -> 时长 -> 评分）
    df = df[['filename', 'len_in_sec', 'MOS_SIG', 'MOS_BAK', 'MOS_OVRL', 'P808_MOS']]
    df.to_csv(args.output_csv, index=False, encoding='utf-8')

      subprocess.run(
        ["npu-smi set -t power-saving -i 0"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 统计信息
    success_count = len(results)
    print(f"\n[处理完成] 成功: {success_count}/{len(audio_files)} | 设备: CANNExecutionProvider")
    print(f"[结果保存] {args.output_csv}（包含: filename, len_in_sec, MOS_SIG, MOS_BAK, MOS_OVRL, P808_MOS）")

if __name__ == "__main__":
    main()