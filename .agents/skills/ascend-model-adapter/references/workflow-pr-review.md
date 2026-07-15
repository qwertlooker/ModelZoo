# ModelZoo PR 审查工作流

## 目录

1. 固定审查对象
2. 获取证据
3. 分层审查
4. Dry run
5. 输出结论

## 1. 固定审查对象

- 记录仓库、base SHA、PR head SHA、merge ref（如可用）和目标模型目录。
- 区分目标仓 PR 与 head fork：评论/流水线 API 使用 PR 所属目标仓；用户给出的 fork URL 只用于定位 head 分支，不能替代目标 PR 标识。
- 重新查询目标仓 `master` HEAD、目标路径及同类近期实质变更样本；不复用历史审查快照。
- 本地检出当前 head；不要只审查 PR 描述或旧评论引用的 commit。
- 多模型 PR 为每个目录建立独立检查项，共享文件另列。

## 2. 获取证据

1. 用 `gitcode_pr_review_sampler.py --pr-path PR=PATH` 验证 PR diff 确实触达目标目录。
2. 若一个 PR 修改多个模型，为同一 PR 重复提供每个精确路径。
3. 区分人工评论、AI review、CI 状态和系统事件；去除 push、改标题、历史流水线噪声。
4. AI review 失败、评论抓取失败或 path 未验证只能标记为证据缺口。
5. 历史 CI 失败用于发现常见问题；当前是否通过只看 head SHA 对应的最新流水线。
6. 核对 discussion `fetched/total`；截断、缺页或超过采样上限时 fail closed。按 `created_at` 查看最新记录，不依赖 API 默认返回的前 20 条。
7. 每次 push 后重新查询 PR head，记录提交事件时间；只采用该事件之后启动并进入终态的门禁结果。`running`、删除旧失败标签或 virtual merge 成功均不是 CI 通过。

## 3. 分层审查

按顺序审查，先报阻塞问题：

1. **范围与完整性**：变更目录、ModeList、冲突标记、误删文件、上游 commit、license、贡献门禁和目标路径已有文件处理。
   - 数据准备章节：来源、目录结构、获取命令；精度/性能表引用的类别名与
     目录结构对应。
   - 公网地址声明列出实际 URL 清单，非泛化描述。
   - README 代码块命令可直接执行：占位符已由前文环境变量定义，无手动替换项。
2. **执行正确性**：语法、未定义参数/变量、拼写、输出数量分支、CLI 多值参数与 flag 顺序、子进程退出码、原生推理会话生命周期。
3. **Patch 与下载**：patch 路径、cwd 变化、非零退出、原始文件 URL、哈希/大小、生成配置来源。
4. **NPU 正确性**：设备默认值、异步计时同步、unsupported op、dtype、shape、CPU fallback。
5. **精度与性能**：数据集/split、metric、完整结果、计时边界、至少 3 次性能与中位数、RTF/FPS/QPS 单位、硬件型号。
6. **文档与候选边界**：工作区 README/NPU_ADAPTATION/ACCEPTANCE_PLAN 职责一致；ACL 候选 README 自包含且只引用候选文件。
7. **CI/合规**：CodeCheck、SCA、Antipoison、license 和复制代码来源。

具体检查点见 `pr-review-heuristics.md`。

## 4. Dry run

先运行 strict quickcheck，再补模型特定检查：

- patch：干净上游 commit 上 `git apply --check`；site-packages patch 用 `find_spec()` 定位、从 site-packages 根目录 `patch -p1` dry run。
- CLI：每个 Python 入口执行 `--help`；构造多值参数、可选 flag、缺参和错误路径。
- 下载/数据：用最小 tar/zip、短音频/小图和最小 label 做 fixture；验证顶层目录展开。
- wrapper：故意让子进程失败，确认父脚本返回非零。
- runtime：对 ais_bench/ACL 等原生会话验证正常、异常和部分构造失败路径均释放；ONNX Runtime 无公共 close API 时在 `finally`/上下文退出清空最后持有引用，不调用私有 `_sess`。
- benchmark：拒绝 loop/record 为 0；确认 NPU 同步和统计边界。
- Markdown：检查内部锚点与真实 heading slug 一致、代码块语言、末尾换行和本地链接；push 后等待当前 head 的文档门禁终态。
- NPU：CPU-only dry run 不能替代真实精度与性能。
- 仓库门禁：只运行当前目标 checkout 自带的 `tools/audit_model_delivery.py`；不要借用另一个 clone 中按 `__file__` 推导 repo root 的脚本。脚本缺失时记录未执行。声明上库候选就绪时追加 target-readiness，并从独立候选目录 clean-room 重放。
- README 可执行性：用 `audit_readme.py` 提取所有 bash 代码块，检查 `<...>`
  占位符是否在前文有 export 定义；逐条验证 cd、python3、atc 等命令引用的
  路径与变量一致；检查公网地址声明是否列出所有实际 URL；检查数据准备章节
  是否覆盖精度表引用的类别。

## 5. 输出结论

每个 finding 包含：严重级别、文件/行、触发条件、影响、证据和最小修复建议。区分：

- **阻塞**：会导致安装、patch、推理、精度、性能或 CI 结论失真。
- **应修复**：可复现性、错误传播、参数边界或维护性问题。
- **建议**：不影响当前结果的清理与风格问题。

最后列出已执行/未执行的检查、当前 head SHA、真实 S0–S4、`target_ready`、NPU 验证状态和剩余证据缺口。S3/S4、上库候选就绪和正式合入分别判断。
