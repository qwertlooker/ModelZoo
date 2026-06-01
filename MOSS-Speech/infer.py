"""MOSS-Speech single-request inference entrypoint for CPU/NPU validation."""

import argparse
from dataclasses import astuple
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig, StoppingCriteria, StoppingCriteriaList


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOSS-Speech generation on a selected device.")
    parser.add_argument("--model", default="fnlp/MOSS-Speech", help="ModelScope/Hugging Face id or local MOSS-Speech directory.")
    parser.add_argument("--codec", default="fnlp/MOSS-Speech-Codec", help="Codec id or local MOSS-Speech-Codec directory.")
    parser.add_argument("--space_dir", default="MOSS-Speech/upstream", help="Local HF Space source directory containing cosyvoice/ and optional Matcha-TTS/.")
    parser.add_argument("--matcha_dir", default=None, help="Optional Matcha-TTS directory if it is not under --space_dir/Matcha-TTS.")
    parser.add_argument("--prompt", default="Hello!", help="User text prompt.")
    parser.add_argument("--system_prompt", default="You are a helpful voice assistant. Answer the user's questions with spoken responses.")
    parser.add_argument("--prompt_audio", default="MOSS-Speech/upstream/assets/prompt_cn.wav", help="Decoder prompt wav for audio generation.")
    parser.add_argument("--output_dir", default="MOSS-Speech/outputs", help="Directory for generated audio/text outputs.")
    parser.add_argument("--output_modality", choices=["audio", "text"], default="audio", help="Generate speech audio or text response.")
    parser.add_argument("--device", choices=["npu", "cpu", "cuda"], default="npu", help="Execution device. Card index is controlled by environment variables.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    return parser.parse_args()


def register_device_backend(device_name: str) -> torch.device:
    if device_name == "npu":
        import torch_npu  # noqa: F401  # Registers the NPU backend.
    return torch.device(device_name)


def add_space_paths(space_dir: str, matcha_dir: str | None) -> None:
    import sys

    space_path = Path(space_dir).resolve()
    sys.path.insert(0, str(space_path))
    matcha_path = Path(matcha_dir).resolve() if matcha_dir else space_path / "Matcha-TTS"
    sys.path.insert(0, str(matcha_path))


class StopOnToken(StoppingCriteria):
    """Stop generation once the last token equals the configured stop id."""

    def __init__(self, stop_id: int) -> None:
        super().__init__()
        self.stop_id = stop_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        return input_ids[0, -1].item() == self.stop_id


def build_stopping_criteria(processor: AutoProcessor) -> StoppingCriteriaList:
    tokenizer = processor.tokenizer
    stop_tokens = [
        tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    return StoppingCriteriaList([StopOnToken(token_id) for token_id in stop_tokens])


def main() -> None:
    args = parse_args()
    add_space_paths(args.space_dir, args.matcha_dir)
    device = register_device_backend(args.device)

    output_modality = [args.output_modality]
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=True,
        use_cache=True,
    )
    messages = [[
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]]

    processor = AutoProcessor.from_pretrained(
        args.model,
        codec_path=args.codec,
        device=args.device,
        trust_remote_code=True,
    )
    stopping_criteria = build_stopping_criteria(processor)
    encoded_inputs = processor(messages, output_modality)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device).eval()

    with torch.inference_mode():
        token_ids = model.generate(
            input_ids=encoded_inputs["input_ids"].to(device),
            attention_mask=encoded_inputs["attention_mask"].to(device),
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )

    results = processor.decode(token_ids.to(device), output_modality, decoder_audio_prompt_path=args.prompt_audio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (result, modality) in enumerate(zip(results, output_modality)):
        audio, text, sample_rate = astuple(result)
        if modality == "audio":
            output_file = output_dir / f"audio_{index}.wav"
            torchaudio.save(str(output_file), audio.cpu(), sample_rate)
            print(output_file)
        else:
            output_file = output_dir / f"text_{index}.txt"
            output_file.write_text(text, encoding="utf-8")
            print(text)


if __name__ == "__main__":
    main()
