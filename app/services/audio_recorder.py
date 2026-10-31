
import sounddevice as sd
import soundfile as sf

from pathlib import Path

from app.config import settings


class AudioRecorder:

    def __init__(self):
        self.sample_rate = settings.SAMPLE_RATE
        self.channels = settings.CHANNELS
        self.dtype = settings.AUDIO_DTYPE

        settings.RECORDINGS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def list_microphones(self):
        """Display available audio input devices."""

        devices = sd.query_devices()

        print("\nAvailable audio devices:")

        for index, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(
                    f"{index}: {device['name']}"
                )

    def record(self, duration=None, filename="recording.wav"):
        """Record audio and save it as a WAV file."""

        if duration is None:
            duration = settings.RECORD_DURATION

        output_path = settings.RECORDINGS_DIR / filename

        print(f"\nRecording for {duration} seconds...")
        print("Speak now!")

        try:
            # Check microphone configuration
            sd.check_input_settings(
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype
            )

            # Capture microphone audio
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype
            )

            sd.wait()

            # Save audio as 16-bit PCM WAV
            sf.write(
                str(output_path),
                audio,
                self.sample_rate,
                subtype="PCM_16"
            )

            print(f"\nRecording saved: {output_path}")

            return output_path

        except sd.PortAudioError as error:
            raise RuntimeError(
                f"Microphone recording failed: {error}"
            ) from error
