import time
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from BEATs import BEATs, BEATsConfig
import torchaudio

# load the fine-tuned checkpoints
checkpoint = torch.load('tianjingcheng/beats/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

#device = torch.device('npu:0' if torch.npu.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Using device: {device}')
BEATs_model.to(device)

# predict the classification probability of each class
audio_input_16khz, _ = torchaudio.load("tianjingcheng/NeMo/nvidia/2902-9008-0000_01.wav")
padding_mask = torch.zeros_like(audio_input_16khz).bool()

audio_input_16khz = audio_input_16khz.to(device)
padding_mask = padding_mask.to(device)

start_time = time.time()
print("start_time: ", start_time)
for _ in range(200):
    probs = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

    for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
        top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
        print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')

end_time = time.time()
print("end_time", end_time)
print("end_time - start_time: ", end_time - start_time)
