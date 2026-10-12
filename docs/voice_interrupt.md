# 实时数字人打断：PR #1 / #2 整合说明

## 基线与逐项比较

本补丁直接基于 `main` 的 `7f0365699ab04dd57078c82ea58220c88521cc21`。
两份原 PR 均只比这一共同祖先多一个独立提交，彼此没有包含关系：

- [PR #1](https://github.com/ZZUZSL1024/OpenAvatarChat/pull/1)：`1745c781a9495d8fc3d37cc1eda2a4ad155fc0b5`，9 个文件。
- [PR #2](https://github.com/ZZUZSL1024/OpenAvatarChat/pull/2)：`1a97a2c3aa102d815150ef92d52ccad0e6f9ca07`，5 个文件。

| 文件/环节 | PR #1 | PR #2 | 本补丁 |
| --- | --- | --- | --- |
| `client_handler_base.py` | 向 delegate 暴露整个 session | 注入信号回调 | 基类声明回调绑定，RTC delegate 转发 |
| `session_context.py` | 共享 `interrupting` 布尔量和 speech ID | 无 | 会话独立的 generation + 锁，无消费者清除中断标志 |
| `chat_session.py` | 设置中断标志、开启 VAD | 清 handler / sink 队列、开启 VAD | 原子提升 generation、清输入/分发/输出队列、立即通知每个 handler；丢弃在途旧结果 |
| `client_handler_rtc.py` | 转发信号、清本地输出 | 转发信号 | 提取独立 delegate；跨线程使用 `queue.Queue`；入队、出队均校验轮次 |
| `rtc_stream.py` | 新文本/stop_chat；麦克风峰值阈值 | 新文本/stop_chat | 全部保留；归一化 RMS + 持续时长去抖；保留触发语音的起始片段；RTC 输出前再次校验 |
| CosyVoice handler | 处理下一条输入时检测布尔量；调用 `streaming_complete` | 无 | 信号到达即失效旧 callback、异步 `streaming_cancel`，异常时关闭连接；正常完成仍使用 complete |
| MuseTalk handler | 下一条音频到达时检查并清标志/队列 | 无 | 每会话独立 processor / 队列 / slicer / 输出线程；信号立即清队列，下一轮处理前清 slicer |
| MuseTalk processor | 清内部队列 | 无 | 每级队列及推理批次保留 generation；在途 GPU 结果不能重新污染清空后的队列；旧 EOS 不恢复新轮次 VAD |
| 文档 | `voice_interrupt.md` | `interrupt_behavior.md` | 本文件统一描述配置、语义、测试和限制，避免两套互相覆盖的说明 |

直接顺序合并两份原 PR 无法解决以下问题：消费者各自清除同一个布尔量；后台任务清队列后继续写回；`streaming_complete` 等待剩余音频而不是取消；MuseTalk 全局可变状态跨会话共用；原始 int16 峰值与归一化阈值不匹配。

## 启用方式与兼容范围

`config/chat_with_openai_compatible_bailian_cosyvoice_musetalk.yaml` 已启用：

```yaml
RtcClient:
  enable_interruption: true
  interrupt_threshold: 0.01  # 麦克风归一化 RMS
  interrupt_duration: 0.096  # 持续超过阈值的秒数
```

启动命令保持不变：

```bash
uv run src/demo.py --config config/chat_with_openai_compatible_bailian_cosyvoice_musetalk.yaml
```

覆盖的完整链路是 RTC → Silero VAD → SenseVoice → OpenAI-compatible LLM → Bailian CosyVoice → MuseTalk → RTC。
其他配置默认不开启，保留旧的异步输出行为。LiteAvatar、LAM、Qwen Omni 等额外后台回调在启用前也必须携带不可变 generation，并实现相应的取消/缓冲清理；不要仅修改配置就认为这些链路已通过验证。

## 轮次与并发契约

1. 只有 ChatSession 在 `INTERRUPT` 时提升会话 generation。不同会话拥有不同的锁、队列和轮次。
2. 输入在进入引擎时打标。同步/生成器 handler 的结果继承输入轮次，不能按输出完成时间重新标记。旧生成器退出时关闭，并关闭 OpenAI streaming response。
3. 跨线程/SDK callback 必须在创建时捕获 generation，通过 `context.submit_data(data, generation=origin_generation)` 提交。启用中断的会话拒绝无轮次的后台输出。
4. 清队列和检查后入队共用会话锁，避免 check-then-put 竞态。锁内禁止等待网络/GPU。
5. CosyVoice 取消先解除旧 synthesizer 引用，随后在后台调用 SDK 取消；旧 callback 即使迟到、复用相同 speech ID，也不能输出音频或结束标志。SDK 最低依赖 `dashscope==1.23.1` 已有 `streaming_cancel`。
6. MuseTalk 音频、Whisper、UNet、VAE、合成、输出都保留原 generation。单线程与拆分 UNet/VAE 两种模式均支持。未满批次在收到新轮次数据时丢弃。中断不销毁 processor，也不等待正在执行的 GPU kernel。
7. VAD/ASR/LLM 的输入缓存在各自 handler 线程进入新轮次前重置；旧的 VAD/ASR/EOS 回调只能更新自己的轮次。客户端 `END` 在启用中断时需携带 generation；无轮次/旧轮次 END 被忽略。
8. RTC 文本打断先取消旧轮次，再提交新文本。语音打断检测在回答中以及已有音频待发送时生效，持续噪声也可能触发；依赖客户端回声消除，并可调整阈值。
9. 模型权重仍可共享，但可变 processor 不共享。模型调用使用共享锁，避免多会话同时使用模型临时状态；这可能降低并行吞吐，需要在目标 GPU 上衡量。

## 自动化测试

CPU 测试直接运行生产 ChatSession、RtcStream、RTC delegate、CosyVoice callback 和 MuseTalk 全部工作线程。
仅 UI/RTC 框架基类、GPU/重采样依赖使用测试替身；模型推理和外部 TTS 服务使用可控假实现。PyAV、DataBundle、队列、线程与实际分发路径均真实执行。

```bash
python3.11 -m venv .venv-interruption
.venv-interruption/bin/python -m pip install -r tests/interruption/requirements.txt
.venv-interruption/bin/python -m pytest tests/interruption -q
```

新增 GitHub Actions `Interruption regression tests`，在 PR 和 main/整合分支 push 上运行 Python 3.11 CPU 测试，不需要 API key、GPU 或模型下载。

覆盖：

- RTC 新文本、停止、无效消息；int16 极值/float 麦克风输入、去抖、触发语音起始片段和 VAD 提前恢复时的音频积压。
- 40 次连续中断、跨会话隔离、原子分发与清理、已运行生成器及同步提交、无轮次回调拒绝。
- TTS 真正取消、阻塞在正常完成时取消、相同 speech ID 的迟到 data/complete/error，以及 callback 检查与提交之间发生中断。
- MuseTalk 在 feature、UNet、VAE、合成和组合推理内被打断；两种推理模式的半批次清理；全部内部/外部队列、音视频与结束事件过滤。
- 实际 TTS callback → 引擎 pump → MuseTalk handler/processor/media thread → RTC emit 的组合回归；中断后仅新轮次音频输出。
- VAD 状态/切片重置、旧 END 不覆盖新轮次状态、会话独立的 processor / 队列 / slicer，以及未启用功能时的兼容行为。

## 验证边界与部署检查

测试证明的是应用层 RTC emit 边界不返回已失效轮次的新数据。已经交给 fastrtc/aiortc 编码器、网络或浏览器 jitter/playback buffer 的媒体无法由服务端队列清理撤回；本补丁不宣称浏览器扬声器在零延迟内静音。

本次没有真实 GPU/模型权重、云 TTS 密钥和浏览器媒体会话，因此没有运行真实音质、口型同步、回声消除、网络抖动或性能测试。部署前在目标环境执行：

- 播报长句时分别发送文本、stop_chat、开口打断，确认能继续完整回答新问题；测量浏览器可闻尾音。
- 连续快速打断，并同时运行两个会话，确认一方停止/关闭不影响另一方。
- 分别设置 `multi_thread_inference: true/false`，检查口型同步、吞吐、断线和异常恢复。
- 模拟 TTS/LLM 网络慢响应：应用层旧数据必须被丢弃，但供应商握手或下一块数据返回前，后台资源释放/新请求调度仍可能有延迟。

整合 PR 替代 #1、#2；原提交和分支保留供追溯，不再分别合并，以免回写共享布尔量或覆盖轮次校验。
