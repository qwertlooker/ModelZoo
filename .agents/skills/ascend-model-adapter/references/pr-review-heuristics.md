# PR 检视启发式

## 来源与边界

2026-07-10 使用精确 `PR=ModelZoo path` 映射抽取了 21 个近期 PR、24 个模型目录的讨论；所有样本均由 PR diff 验证路径，共取得 220 条去重前后可分析记录。只把这些评论用于发现通用缺陷模式；历史 CI、系统事件和 AI review 失败不作为当前 head 的结论。

## 阻塞级检查

- PR head、base、目标目录和本地 checkout 必须一致；评论引用的旧 commit 只作线索。
- 每次重新查询目标 `master` HEAD、拟合入路径和近期实质样本；`NPU_ADAPTATION.md` 必须记录日期/commit/参考目录/贡献门禁/候选清单。
- 冲突标记、语法错误、未定义变量/参数、错误脚本名、下载到 HTML、patch 失败却退出 0 都会直接破坏复现。
- wrapper 中的 `subprocess.run` 必须检查返回码或使用 `check=True`；推理/转换子进程失败时父进程不能返回成功。
- patch helper 在改变 cwd 前把 patch 路径解析成绝对路径；文件不存在、dry run 失败、应用失败均返回非零。
- site-packages patch 用 `find_spec()` 定位且从 site-packages 根目录 `patch -p1`；不能 import 包只为取得 `__file__`。
- 用户可见下载命令验证 URL 是原始文件端点；Hugging Face 文件不能误用返回网页的 blob 地址。重要权重/配置记录大小或 SHA256。
- README 中每个 requirements、patch、脚本、配置和数据文件名与仓内实际大小写完全一致。

## CLI 和代码正确性

- 运行 `py_compile`、lint 和 `--help`，并检查拼写、缩进、未定义 args、未定义输出、残留 import 和 debug code。
- 对 `--traj-ids 0 1 --flag value` 等多值参数构造回归用例，确保解析循环不会吞掉后续 flag。
- 为输出数量、空输出、异常 shape 和未来新增输出写显式分支；不要只覆盖当前常见 `len(outputs)`。
- 参数拒绝零/负 loop、record、batch 和空数据集，避免除零或假成功。
- 公共 helper 去重；推理和评测重复逻辑提取到实际提交的模块，并检查删除旧模块后的引用。
- ais_bench/ACL 等持有设备资源的会话必须在上下文管理器或 `finally` 中调用公开释放接口（如 `free_resource()`）；覆盖正常退出、业务异常和第二个会话构造失败等部分初始化路径。
- ONNX Runtime `InferenceSession` 没有公共 close API 时，作用域退出必须清空最后持有引用；不要调用私有 `_sess`/`_reset_session`。用 fake session + weakref 或等价夹具验证引用释放，同时验证 OM 会话释放次数。
- 删除无必要的 `# noqa`/lint 抑制、临时注释和乱码；确需抑制时写具体理由和范围。

## NPU 正确性与性能

- 推理、评测、benchmark 默认 NPU；物理卡通过 `ASCEND_RT_VISIBLE_DEVICES` 选择。
- 纯模型 NPU 计时在区间前后 synchronize；端到端计时可保留完整流程，但必须标明边界，不能混称纯模型性能。
- 首次编译、cache miss、数据加载、后处理、网络、排队和 CPU fallback 分开记录。
- 性能命令、脚本默认值和表格中的 batch/并发/warmup/loop/单位一致；音频优先 RTF，可同时列来源明确的 RTFx。
- 正式性能至少独立执行 3 次并报告中位数；保留每次结果，CPU 性能不用于推断加速比。
- unsupported op 的范围要按硬件/软件栈说明，不能把某型号或某版本的限制扩大成所有 Ascend 设备结论。
- wrapper 和 benchmark 的失败路径、空输入、OOM/子进程失败至少做一次负向测试。

## 精度、数据与配置来源

- 不能只写“精度持平”；给出数据集/split、样本数、metric、命令和实际结果。
- 生成的归一化统计、label map、manifest 或配置文件必须说明原始数据、生成脚本和命令；原仓没有的文件尤其如此。
- 区分用户需要下载的原始数据、脚本生成的中间文件和仓内自带 fixture；每个路径写清生产者与消费者。
- 环境数据优先 wget/curl + file/size/SHA 校验；Python 专用下载器必须有固定版本离线替代，offline 缺文件时严格失败。
- 多语言/多 split 数据准备参数要与 README 示例一致，并解释为何选择不同数据源。
- 模型标题、配置和权重必须成套；示例使用 SAM2.1 等变体时不能只用 SAM2 泛称而不解释。
- 许可证检查不仅确认文件存在，还要判断模型、权重和数据是否允许当前分发/商用方式。

## README 与上库结构

- 固定上游 commit/revision，并提供干净 clone 的 patch 命令。
- 对外硬件字段使用准确产品型号；不要用过宽或不适用的系列名掩盖实际验证设备。
- 输出路径、ONNX/OM 路径、权重、batch、SOC_VERSION、数据和并发参数对用户暴露；不要把开发期路径写死。
- 数据准备脚本优先单一 Python 入口；以最小 tar/zip/音频/图片/label fixture 验证目录展开和输出清单。
- README 的 next step、脚本打印提示和 PR Self-test 只能引用真实存在的命令/文件。
- README 目录锚点按实际标题 slug 校验，尤其不能漏掉中文字符；同时检查 fenced code language、末尾换行和相对链接。文档门禁通过前不要只凭本地渲染判断。
- 工作区三类主文档职责分离；正式候选 README 自包含，不依赖默认排除的 NPU_ADAPTATION/ACCEPTANCE_PLAN。
- S0–S4 只按真实证据提升；至少 S3 + target-readiness 审计 + 独立候选 clean-room 重放，才能写"上库候选就绪"。
- ModeList 统计、表格空单元格、重复段落和模板占位在提交前清理。
  **ModeList.md 合计数字不能重新从头计数**：必须在远端上游 `master` 当前
  合计数字基础上 +N（N 为本 PR 实际新增模型数）。合并上游/变基后必须重新
  查询上游最新数字并叠加本地增量，禁止回退到旧快照的计数。GPL/built-in+contrib
  两个合计行分别校验。

## README 内容深度检查

以下检查验证"章节内容是否充分"，而非仅"章节是否存在"。

### 命令可执行性（阻塞）
- 提取所有 bash 代码块，逐条检查 `<...>` 占位符：每个占位符必须在前文有
  `export VAR=...` 定义并在后续用 `${VAR}` 引用，或属于 README 明确说明的
  替换规则。
- 禁止需要用户手动猜测路径的占位符，如 `<DNSMOS根目录>`、`<测试音频路径>`。
- 涉及两个目录（上游目录 vs 交付目录）时，README 必须在前文用 export
  明确两者的关系和变量名。
- `--model_dir .` 等相对路径必须与前文 cd 命令一致，且 cd 目标不能是占位符。

### 数据准备完整性（阻塞）
- 精度/性能表引用特定数据集、类别或样本数时，README 必须有数据准备章节。
- 数据准备章节必须覆盖：数据来源（URL 或合成方法）、期望目录结构树、
  获取/生成命令。
- 验证脚本用文件路径派生类别（如 `wav.parent.name`）时，README 必须说明
  目录组织方式，且目录中的类别名与精度表中的类别名一致。
- `your_audio_folder` 等示例占位路径不能作为唯一的数据准备说明。

### 公网地址清单（阻塞）
- 公网地址声明不能只写泛化描述（如"本 README 中的 GitHub、GitCode 地址..."）。
- 必须列出 README 中实际引用的所有 URL 清单（表格或列表），包括源码仓库、
  权重下载、数据下载、文档链接和容器镜像地址。
- 用脚本提取 README 全文所有 URL，与公网地址声明中的 URL 逐一比对。

### 章节内容深度（应修复）
- 精度章节不能只写"精度持平"，必须有数据集/split/样本数/metric/命令/结果。
- 性能章节不能只列单一数值，必须有 warmup/loops/独立执行次数/中位数。

## CI 与审查证据

- 区分 Antipoison、CodeCheck、SCA；流水线总失败时查具体阶段，不凭总状态猜原因。
- 最新 head 的 CI 才决定当前状态；旧 commit 的失败/成功记录不能覆盖当前结果。
- GitCode discussion API 默认页可能只含较早的 20 条记录；正式结论要求 `fetched/total` 完整并按 `created_at` 取最新。每次 push 后等待该 head 之后的新门禁从 running 进入 success/failed，删除旧标签不代表成功。
- AI review 摘要需回到实际 diff 验证；AI review 超时或失败只记"未获得意见"。
- 正式审查报告列出未执行的 NPU 精度/性能测试，不能用评论或截图替代。
