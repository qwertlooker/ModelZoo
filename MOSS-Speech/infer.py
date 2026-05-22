"""MossSpeech inference demo aligned with Hugging Face Transformers guidelines."""
import sys
sys.path.append("./MOSS-Speech-space/MOSS-Speech/")
sys.path.append("./MOSS-Speech-space/MOSS-Speech/Matcha-TTS/")
from torch_npu.contrib import transfer_to_npu
import os
from dataclasses import astuple

import torch
import torchaudio

from modelscope import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


prompt = "Hello!"
prompt_audio = "tianjingcheng/openmoss/MOSS-Speech-space/MOSS-Speech/assets/prompt_cn.wav"
model_path = "tianjingcheng/openmoss/MOSS-Speech"
codec_path = "tianjingcheng/openmoss/MOSS-Speech-Codec"
output_path = "outputs"
output_modality = ["audio"]  # or text

generation_config = GenerationConfig(
    temperature=0.7,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    max_new_tokens=1000,
    min_new_tokens=10,
    do_sample=True,
    use_cache=True,
)


class StopOnToken(StoppingCriteria):
    """Stop generation once the final token equals the provided stop ID."""

    def __init__(self, stop_id: int) -> None:
        super().__init__()
        self.stop_id = stop_id

    def __call__(self, input_ids: torch.LongTensor, scores) -> bool:  # type: ignore[override]
        return input_ids[0, -1].item() == self.stop_id


def prepare_stopping_criteria(processor):
    tokenizer = processor.tokenizer
    stop_tokens = [
        tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    return StoppingCriteriaList([StopOnToken(token_id) for token_id in stop_tokens])


messages = [
    [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Answer the user's questions with spoken responses."},
            # "content": "You are a helpful assistant. Answer the user's questions with text."}, # if output_modality = "text"
        {
            "role": "user",
            "content": prompt
        }
    ]
]


processor = AutoProcessor.from_pretrained(model_path, codec_path=codec_path, device="cuda", trust_remote_code=True)
stopping_criteria = prepare_stopping_criteria(processor)
encoded_inputs = processor(messages, output_modality)

model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="cuda").eval()

with torch.inference_mode():
    token_ids = model.generate(
        input_ids=encoded_inputs["input_ids"].to("cuda"),
        attention_mask=encoded_inputs["attention_mask"].to("cuda"),
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    )

results = processor.decode(token_ids, output_modality, decoder_audio_prompt_path=prompt_audio)

os.makedirs(output_path, exist_ok=True)
for index, (result, modality) in enumerate(zip(results, output_modality)):
    audio, text, sample_rate = astuple(result)
    if modality == "audio":
        torchaudio.save(f"{output_path}/audio_{index}.wav", audio, sample_rate)
    else:
        print(text)
