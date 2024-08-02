import librosa as librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os

def main(data_dir):
    for file in os.listdir(data_dir)[1:]:
        new_dir = data_dir + file + "/"
        if len(os.listdir(new_dir)) < 20:
            for i in os.listdir(new_dir):
                if i.endswith(".wav"):
                    for num in range(2):
                        file_path = os.path.join(new_dir, i)
                        augmented_audio, sr = audio_augmentation(file_path)
                        output_path = os.path.join(new_dir, f"augmented_{num}{i}")
                        audio_save(augmented_audio, sr, output_path)
        if len(os.listdir(new_dir)) < 30:
            for i in os.listdir(new_dir):
                if i.endswith(".wav"):
                    file_path = os.path.join(new_dir, i)
                    augmented_audio, sr = audio_augmentation(file_path)
                    output_path = os.path.join(new_dir, f"augmented_{i}")
                    audio_save(augmented_audio, sr, output_path)
        for wav_file in os.listdir(new_dir):
            if wav_file.endswith(".wav"):
                audio_path = os.path.join(new_dir, wav_file)
                y, sr = librosa.load(audio_path, sr=None)

                generating_mel_specs(audio_path, specs_dir, wav_file, file)


def audio_save(audio_file, sr, file_path):
    sf.write(file_path, audio_file, sr)

def audio_augmentation(file_name):
    y, sr = librosa.load(file_name, sr=None)

    stretch_factor = np.random.uniform(low=0.8, high=1.2)
    audio_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)

    pitch_shift = np.random.randint(-4, 5)
    audio_pitch_shift = librosa.effects.pitch_shift(audio_stretched, sr=sr, n_steps=pitch_shift)

    noise = np.random.randn(len(audio_pitch_shift)) * 0.005
    audio_noisy = audio_pitch_shift + noise

    shift_samples = np.random.randint(-int(sr * 0.1), int(sr * 0.1))
    audio_shifted = np.roll(audio_noisy, shift_samples)

    return audio_shifted, sr

def generating_mel_specs(audio_file, specs_dir, wav_file, file):

    y, sr = librosa.load(audio_file, sr=None)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-Spec of {wav_file}")
    plt.tight_layout()
    make_new_dir = os.path.join(specs_dir, file)
    if not os.path.exists(make_new_dir):
        os.makedirs(make_new_dir)
    spec_path = os.path.join(make_new_dir, f"{wav_file}.png")
    plt.savefig(spec_path)
    plt.close()

data_dir = "./data/"
specs_dir = "./mel_specs/"

if __name__ == "__main__":
    main(data_dir)
