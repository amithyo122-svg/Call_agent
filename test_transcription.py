
import json

from app.config import settings
from app.models.moss_transcriber import MossTranscriber


def main():

    audio_path = settings.RECORDINGS_DIR / "test_recording.wav"

    if not audio_path.exists():
        print("Audio file not found!")
        return

    # Initialize model
    transcriber = MossTranscriber()

    # Convert speech to text
    result = transcriber.transcribe(audio_path)

    # Display transcript
    print("\n========== TRANSCRIPTION ==========\n")

    print(result["text"])

    print("\n========== SPEAKER SEGMENTS ==========\n")

    for segment in result["segments"]:

        print(
            f"[{segment['start']:.2f}s - "
            f"{segment['end']:.2f}s] "
            f"{segment['speaker']}: "
            f"{segment['text']}"
        )

    # Save results
    settings.TRANSCRIPTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        settings.TRANSCRIPTS_DIR / "transcription.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        settings.TRANSCRIPTS_DIR / "transcription.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(result["text"])

    print("\nTranscription saved successfully!")


if __name__ == "__main__":
    main()
