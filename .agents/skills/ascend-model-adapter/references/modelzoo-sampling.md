# ModelZoo `ACL_PyTorch/built-in` sampling guidance

Snapshot source: `https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in` and category pages, checked on 2026-06-30. The repository currently exposes these built-in categories: `audio`, `cv`, `nlp`, `ocr`, `embedding`, `foundation_models`, and `embodied_ai`. Parsed visible model counts, excluding helper/hidden files: audio 27, cv 90, nlp 17, ocr 9, embedding 4, foundation_models 6, embodied_ai 4.

When adapting a new model, refresh this list with `scripts/modelzoo_sampler.py` and prefer newer merged projects because README style, images, CANN versions, and validation expectations drift quickly.

## Recent representative sample set

This sample intentionally over-covers “about 20” projects so every type is represented and recent merges dominate.

| Type | Project | Recent page signal | Why it is useful |
|---|---|---:|---|
| embodied_ai | GraspNet | 22 小时前, PR 7630 | Latest embodied OM path; custom pointnet/point-cloud utilities, install script, eval/infer split. |
| cv | InstantID | 3 天前, PR 7624 | Recent complex CV/VLM-ish pipeline; multiple patches, ONNX shape fixes, MagicONNX, ais_bench components. |
| cv | PromptIR | 6 天前, PR 7623 | Recent image restoration; image-first README, patch workflow, dependency pinning. |
| audio | YingMusic-SVC_for_Pytorch | 7 天前, PR 7609 | Recent audio/SVC; strong image-first README, offline weight guidance, torch_npu mismatch FAQ. |
| cv | F3Net | 13 天前, PR 7610 | Recent CV saliency; image container contract, accuracy/performance scripts, CPU-vs-NPU metrics. |
| cv | SAM2 | 19 天前, PR 7613 | Recent segmentation optimization pattern; useful if available during refresh. |
| embodied_ai | IsaacGR00T | 19 天前, PR 7616 | Torch/robotics model; patch plus NPU utility, HF download, environment-heavy setup. |
| cv | SAM3 | 22 天前, PR 7604 | Modern foundation CV; ONNX export/optimize, FlashAttentionTik patch, conversion shell, coco IoU eval. |
| cv | FocalFormer3D_for_Pytorch | 21 天前, PR 7595 | 3D detection; large dataset notes, DrivingSDK/custom deps, image-first style. |
| nlp | chronos-2 | 21 天前, PR 7581 | Recent NLP/time-series; direct ascend inference plus performance/accuracy scripts. |
| audio | Canary-1B | 28 天前, PR 7592 | Recent ASR/AST; CANN 8.5.1 + torch_npu 2.9.0 style, dataset preparation, RTFx/WER/BLEU. |
| embodied_ai | vla/pi0 | 30 天前, PR 7590 | Decomposed VLA model; separate ONNX/OM verification scripts for VLM and action expert. |
| audio | Index-TTS-vLLM-v2 | 1 个月前, PR 7579 | vLLM-Ascend TTS service route, FastAPI serving, environment variables, RTF. |
| cv | D-FINE | 1 个月前, PR 7573 | Detection model; patch, ONNX/OM, `om_inf.py`, ais_bench performance. |
| audio | CosyVoice3 | 1 个月前, PR 7565 | TorchAir/vLLM image route; docker launch and service-style inference. |
| nlp | ProtBert_for_Pytorch | 1 个月前, PR 7569 | Classic Hugging Face encoder; `TestProtbert_2onnx.py`, static shape ATC, `ais_bench`. |
| ocr | PP-DocLayoutV2 | 1 个月前, PR 7594 | Paddle/PaddleX + ONNX/OM; msit surgeon, dynamic batch, downstream OCR/VLM accuracy. |
| ocr | PP-DocLayoutV3 | 1 个月前, PR 7594 | Paddle layout detector; dynamic input shape ATC and demo inference. |
| ocr | PaddleOCR-VL-1.5 | 1 个月前, PR 7594 | vLLM-Ascend VLM service and OmniDocBench end-to-end evaluation. |
| ocr | UVDoc | 1 个月前, PR 7533 | OM route with MagicONNX patching, tesseract accuracy evaluation, custom benchmark. |
| embedding | bge-m3 | 1 个月前, PR 7587 | TorchAir embedding route; HF model clone, simple `infer.py`, NPU ID parameter. |
| embedding | bge-reranker-v2-m3 | 1 个月前, PR 7587 | Reranker variant; similar route but ranking-score semantics. |
| foundation_models | Chinese_CLIP | 1 个月前, PR 7537 | Dual encoder export/OM/eval shell pattern; patch and CLIP retrieval metrics. |
| foundation_models | SigLIP2 | 2 个月前, PR 7517 | Dual text/vision ONNX→OM models; preprocess/postprocess, ImageNet accuracy, ais_bench. |

## Patterns to copy

- Newer projects often place the ModelZoo directory as the authoritative delivery bundle and instruct users to clone upstream source at a fixed commit, then apply `diff.patch` or model-specific patch files.
- README headings usually follow: overview, input/output data, inference environment, quick start, get source, install dependencies, prepare weights/data, export model, convert OM, run inference, accuracy, performance, FAQ/known issues.
- Recent image-first READMEs include a “version declaration” table and warn not to reinstall image-provided `torch`/`torch_npu`.
- OM projects commonly include `export_onnx.py`/`pth2onnx.py`, optional ONNX fix/optimization scripts, `convert_om.sh` or embedded `atc` commands, `infer.py`, `eval_accuracy.py`, and `eval_performance.py`/`benchmark.sh`.
- vLLM/TorchAir projects document container launch, service command, NPU memory/env variables, and task-specific client scripts rather than ATC conversion.
- Accuracy evidence may be numeric tensor diff, task metrics (WER/BLEU/mAP/IoU/overall), or end-to-end service evaluation; performance evidence must state chip, batch/concurrency, input shape, precision, warmup/loop count, and tool.
