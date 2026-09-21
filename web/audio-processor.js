/**
 * AudioWorklet 处理器：把麦克风采样累积成固定长度（默认 600ms @16kHz = 9600 点）的块
 * 每凑满一块就 postMessage 给主线程，由主线程转 PCM16 后经 WebSocket 发送
 */
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 每 600ms 发送一次；用实际采样率计算，避免设备采样率不是 16k 时块长不对
        this.bufferSize = Math.round(sampleRate * 0.6) || 9600;
        this.buffer = new Float32Array(this.bufferSize);
        this.offset = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];
        for (let i = 0; i < channelData.length; i++) {
            this.buffer[this.offset] = channelData[i];
            this.offset++;

            if (this.offset >= this.bufferSize) {
                // 发送完整缓冲区（slice 复制一份，避免后续被覆写）
                this.port.postMessage(this.buffer.slice(0));
                this.offset = 0;
            }
        }
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
