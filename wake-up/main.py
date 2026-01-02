import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

openwakeword.utils.download_models()

SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms

model = Model(
    wakeword_models=["alexa"],
    inference_framework="onnx"
)

pa = pyaudio.PyAudio()

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("🎙️ Listening...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)

        predictions = model.predict(audio)

        for wakeword, score in predictions.items():
            if score > 0.5:
                print(f"🔥 Wake word detected: {wakeword} ({score:.2f})")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.close()
    pa.terminate()
