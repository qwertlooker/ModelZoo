# Project Constraints

- After making requested changes in this repository, commit them to the current branch by default.
- After committing, push the current branch to the configured remote by default.
- Do not force-push or rewrite published history unless explicitly requested.
- If a push fails because authentication, permissions, network, or remote divergence blocks it, report the exact failure and leave the local commit intact.

# Project-wide Adaptation Script Standards

- These standards apply to every model directory in ModelZoo, not only Canary-1B.
- For adaptation, evaluation, and data-preparation scripts, keep required dependencies as top-level imports by default. Device backend registration modules may be conditional, e.g. import `torch_npu` only on the NPU path.
- Do not add unnecessary compatibility layers or silent fallbacks. Missing dependencies, missing official fields, incompatible upstream versions, or unavailable official evaluation components should fail promptly and expose the original error.
- Do not replace official evaluation paths with regex/basic normalizers, similarly named third-party packages, simplified metrics, CPU fallbacks, or remote download fallbacks unless explicitly documented and validated as a separate non-official mode.
- Follow `模型NPU 适配标准流程.md` as the project-level workflow source of truth.
