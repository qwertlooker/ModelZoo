# Ascend 适配路线选择

## 决策前探针

先记录：输入是否动态、是否有 Python control flow、多输入输出/cache、custom op、模型规模、目标延迟/吞吐、服务协议、同任务近期 ModelZoo 路线和目标软件栈支持情况。

## 路线矩阵

| 路线 | 优先条件 | 必做探针 | 主要风险 | 切换条件 |
|---|---|---|---|---|
| ONNX → ATC → OM | 输入 shape 可控、离线部署、需要 ais_bench/ACL | 最小 ONNX export、checker、ATC 小 shape | 动态轴、control flow、unsupported op、external data | 导出语义破坏或 ATC 无法稳定支持时转 TorchAir/拆图 |
| torch_npu eager | 上游 PyTorch 路径完整、先验证正确性或样本已采用 | `.to('npu')` 单样例、算子/dtype 检查 | 性能不足、Python 开销 | 正确性通过后按性能目标转 TorchAir/融合算子 |
| TorchAir | 动态输入/control flow 强、需要图编译且 ONNX 脆弱 | 最小图编译、动态输入、cache 冷热启动 | 首次编译、graph break、缓存失效 | 图切分不稳定时局部 eager 或 hybrid |
| vLLM-Ascend | LLM/VLM/TTS 服务依赖调度、paged attention 或 OpenAI API | server health、单请求、并发、KV cache | 版本配套、显存、调度/模型支持 | 模型未被支持时 TorchAir/torch_npu 服务或 hybrid |
| hybrid | pipeline、多子模型、部分 custom op/CPU 算法 | 每个子图 I/O 与数值对齐 | 跨设备拷贝、端到端口径混淆 | CPU fallback 成为瓶颈时继续迁移对应组件 |

## 选择规则

- 不以“模板有脚本”作为路线依据；至少完成一个路线特定探针。
- 多组件模型先画组件表，再为每个组件单独选路线；总路线记为 `hybrid`。
- CPU fallback 必须写具体阻塞、输入输出边界和耗时占比，不能只写“上游默认 CPU”。
- 使用者提供 checkpoint 时围绕该 checkpoint 选路线，不为迁就路线静默换权重。
- 路线变更要在项目日志记录失败探针、根因和新路线，不删除旧证据。
- 工具版本按路线记录：只有 ONNX/OM 强制 ATC/ais_bench/msit；TorchAir/vLLM 记录各自组件，不强制无关工具。
