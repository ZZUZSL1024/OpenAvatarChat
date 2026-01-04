# 实时语音打断功能说明

本次更新为 MuseTalk + CosyVoice 实时语音对话链路增加了可中断能力，当数字人正在播报时，新的用户问题会立即终止当前回复并开始新的识别/生成流程。

## 功能逻辑
- **中断信号**：
  - WebRTC 数据通道收到 `stop_chat` 或新的 `chat` 文本消息时，会向引擎发送 `interrupt` 信号并立即清空客户端缓存。
  - 麦克风输入在数字人说话期间出现明显音量（默认阈值 0.01）时，也会自动触发中断，重新开启 VAD。
- **引擎共享状态**：新增 `shared_states.interrupting`、`current_speech_id` 用于标记当前播报是否需要终止，以及对齐最新的播报轮次。
- **流水线响应**：
  - CosyVoice TTS 在收到中断时会停止正在进行的流式合成并丢弃旧的音频片段。
  - MuseTalk 数字人处理器会清空内部/输出队列，丢弃旧的音视频帧，并恢复到可监听状态。
  - ChatSession 在收到中断信号后立即重新允许 VAD 检测，从而接受新的用户问题。

## 使用方法
1. 正常启动服务：
   ```bash
   uv run src/demo.py --config config/chat_with_openai_compatible_bailian_cosyvoice_musetalk.yaml
   ```
2. 在前端页面：
   - 当需要打断当前回复时，可在文本输入框发送新的消息或调用前端的“停止”按钮（会发送 `stop_chat`）。
   - 直接开口说出新问题也会自动触发打断，数字人会立即停止播报并重新开始倾听。
3. 若需要调整语音触发阈值，可修改 `src/service/rtc_service/rtc_stream.py` 中 `receive` 方法的 `0.01` 阈值。

## 注意事项
- 打断时会清空当前回复的音频/视频输出队列，避免旧内容在打断后继续播放。
- 中断只影响当前轮次的播报，新的提问会生成新的 `speech_id`，并完整走一遍 ASR/LLM/TTS/Avatar 链路。
