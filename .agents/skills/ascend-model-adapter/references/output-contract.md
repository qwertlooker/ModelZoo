# ModelZoo 交付契约

## 目录

1. 工作区与正式候选目录
2. README 信息
3. 容器命令模板
4. 上库自检
5. PR 描述

## 1. 工作区与正式候选目录

先区分两层：

- **适配工作区**默认维护 `README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`。后两者记录内部事实与验收，不替代 README。
- **正式 ACL 候选目录**按 `NPU_ADAPTATION.md` 的候选清单组装到 `ACL_PyTorch/built-in/<category>/<model>` 风格扁平目录；默认排除内部两份文档、日志、upstream 和 cache。

```text
<ModelName>/
├── README.md
├── requirements.txt
├── diff.patch 或 <model>_NPU.patch     # 修改上游时
└── 按实际路线保留的少量文件
    ├── export_onnx.py / pth2onnx.py
    ├── convert_om.sh / atc.sh
    ├── infer.py / eval_accuracy.py      # 上游没有对应入口时
    ├── benchmark.py / benchmark.sh
    └── prepare_data.py
```

当前贡献规范若要求许可证文件、`modelzoo_level.txt`、自测试入口或其他元数据，按实时目标仓规则加入候选；不能因历史目录缺失而跳过，也不能生成未执行的虚假文件。

- 优先 patch 上游已有入口，不复制重复脚本。
- `ModeList.md` 合计数字不要重新计数；在远端上游当前数字基础上 +1（或按实际新增模型数叠加）。合并上游后同样先取上游最新数字再加本地增量，不能回退到旧计数。
- `requirements.txt` 只放业务依赖；不要重装镜像配套的 torch/torch_npu/vision/audio。
- 数据准备默认一个 Python 主入口；shell 只用于必要系统工具编排。
- `.ascend-adaptation`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、临时 clone、cache、权重、数据和大体积日志默认不进入 ACL 候选。
- 除非用户明确要求，不修改原始 `README_old.md`；其内容需要迁移时无损收敛到三类主文档，再按候选清单排除。
- `README.md` 必须自包含，不能要求 ACL 用户读取被排除的内部文档。
- `NPU_ADAPTATION.md` 记录目标仓快照/路径、版本边界、适配事实、验证命令、S0–S4、候选清单和排除项。
- `ACCEPTANCE_PLAN.md` 记录官方/公开基线、数据/evaluator、功能/L2/扩展验收、阈值、最低正式清单和实际结果。

## 2. README 信息

章节名称跟随同任务近期样本，但必须覆盖：

1. 模型任务、上游 URL/commit、license、权重/配置和适配路线。
2. 实测对外硬件型号、镜像/CANN/Python/torch_npu 及路线相关工具版本。
3. 容器启动、环境检查和工作目录。
4. 源码、patch、依赖安装；写明子包和顶层 `--no-deps` 顺序。
5. 权重、数据、评测工具、protocol/label 的来源、目录树和生成命令。
6. 路线命令：导出/ATC、TorchAir 编译或 vLLM server/client。
7. NPU 推理、输入输出 contract、默认参数和结果路径。
8. 精度与性能：官方/GPU 参考来源、NPU 实测、命令、完整口径和差异；正式性能至少 3 次并报告中位数。
9. FAQ：只放用户复现需要知道的依赖、cache、长编译、unsupported op 和 CPU fallback。
10. README 实际引用的公网地址。

不要把 TODO、内部路径、未提交脚本、理论结果或 CPU 性能带入最终 README。没有真实 NPU 证据时标记 `待 NPU 验证`。

## 3. 容器命令模板

按用户要求保留该模板；使用时根据目标机器和实际权限调整设备、挂载与缓存目录。不要把不需要的宿主机凭据或缓存带入处理不可信仓库的容器。

```bash
export IMAGE=<ascend-image-tag>
docker pull ${IMAGE}

docker run -itd -u root --net=host --privileged=true \
  --name <container-name> \
  --shm-size=256g \
  --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v <宿主机工程目录>:<宿主机工程目录> \
  -v /root/.cache:/root/.cache \
  ${IMAGE} bash -i

docker exec -it <container-name> bash
cd <宿主机工程目录>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

多卡时增加实际需要的 `/dev/davinci*`。镜像 tag 与目标硬件/CANN 栈必须成套；能获取 digest 时把 digest 记入项目证据。

## 4. 上库自检

- [ ] 上游 commit、license、权重/config/tokenizer/label map 已固定。
- [ ] 已重新查询目标 master HEAD、目标路径和近期实质样本，并写入 `NPU_ADAPTATION.md`。
- [ ] patch 在干净上游通过 check；helper 失败返回非零。
- [ ] 所有 README 文件名、路径、URL、参数和 next step 已验证。
- [ ] 下载命令得到原始文件，重要 artifact 有大小或 SHA256。
- [ ] requirements 未覆盖镜像核心栈，import/`--help`/单样例通过。
- [ ] 推理、评测、benchmark 默认 NPU；CPU fallback 有具体边界。
- [ ] 精度使用完整声明数据集/split；性能计时边界和 NPU 同步正确。
- [ ] README、脚本输出、项目日志和 PR 描述中的结果一致。
- [ ] 对外硬件型号准确；未泄露内部代号或详细芯片信息。
- [ ] 交付脚本无 `sys.exit()` / 分号多语句 / 算术操作符缺空格等编码规范问题；已通过 `ruff check` 或等价静态检查零错误。
- [ ] strict quickcheck、ModeList、CodeCheck/SCA/Antipoison 预检完成。
- [ ] 工作区三类主文档职责分离且相互一致，README 不依赖内部证据。
- [ ] 已运行 `tools/audit_model_delivery.py` 基础审计；上库时运行 target-readiness。
- [ ] 已从独立候选目录完成 clean-room 最低路径重放。
- [ ] 技术状态真实达到 S3/S4，并单独满足 `target_ready=true`；否则明确列出未完成门禁。

## 5. PR 描述

使用 Motivation、Modification、Self-test、BC-breaking、Checklist 五段。Self-test 至少列：

| 项目 | 命令/环境 | 结果 | 证据 |
|---|---|---|---|
| 上游/patch | commit；`git apply --check` | PASS/TODO | 日志 |
| 依赖/接口 | import；`--help`；负向参数 | PASS/TODO | 日志 |
| 转换/服务 | export/ATC/TorchAir/vLLM | PASS/TODO | artifact hash |
| NPU 推理 | 单样例命令 | PASS/TODO | 输出/log |
| 精度 | 数据集/split/metric | 结果/TODO | 评测日志 |
| 性能 | 硬件、batch、warmup、loop、3 次结果/中位数、口径 | 结果/TODO | benchmark 日志 |

TODO 只能表示尚未完成的真实门禁，不能与“可上库”或“验证通过”同时出现。
