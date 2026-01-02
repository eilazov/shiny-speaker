import whisper

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    print("▶️ Loading model:", model_size)
    model = whisper.load_model(model_size)

    print("▶️ Transcribing:", audio_path)
    result = model.transcribe(audio_path)

    print("▶️ Raw result:")
    print(result)

    return result.get("text", "")

if __name__ == "__main__":
    print("🚀 Script started")
    text = transcribe_audio("test_audio.m4a")
    print("📝 FINAL TEXT:")
    print(text)
