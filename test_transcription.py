from app.config import settings
from app.models.indic_transcriber import IndicTranscriber


def main():

    audio_path = settings.RECORDINGS_DIR / "test_recording.wav"

    if not audio_path.exists():
        print("Audio file not found!")
        return

    transcriber = IndicTranscriber()

    result = transcriber.transcribe(audio_path)

    print("\n========== TRANSCRIPTION ==========\n")
    print(result)

    settings.TRANSCRIPTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = settings.TRANSCRIPTS_DIR / "transcription.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(result)

    print(f"\nTranscript saved to: {output_file}")


if __name__ == "__main__":
    main()
