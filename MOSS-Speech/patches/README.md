# MOSS-Speech upstream patches

Current status: no upstream source patch is committed in this adaptation directory.

The local `MOSS-Speech/infer.py` is an adaptation-side entrypoint and is not part of the upstream patch set. The Hugging Face Space source was cloned to `MOSS-Speech/upstream/` for inspection at commit `92a89018a8aa6b36f08c366c2659c76ffdc3f980`.

If later validation proves that upstream files must be changed, modify `MOSS-Speech/upstream/` and generate patch files here, for example:

```bash
git -C MOSS-Speech/upstream diff -- cosyvoice/hifigan/generator.py > MOSS-Speech/patches/0001-npu-hifigan-istft.patch
git -C MOSS-Speech/upstream apply --check ../patches/0001-npu-hifigan-istft.patch
```

Do not patch installed third-party packages in-place. Any required `diffusers` / `transformers` change must be bound to an exact upstream version and recorded as a reproducible patch or rejected during validation.
