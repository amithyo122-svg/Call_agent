
from app.services.audio_recorder import AudioRecorder


def main():

    recorder = AudioRecorder()

    # Display microphone devices
    recorder.list_microphones()

    # Record 10 seconds of audio
    audio_path = recorder.record(
        duration=10,
        filename="test_recording.wav"
    )

    print("\nAudio recording completed!")
    print(f"File location: {audio_path}")


if __name__ == "__main__":
    main()
