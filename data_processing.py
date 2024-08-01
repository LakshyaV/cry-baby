import librosa as librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def generating_mel_specs(data_dir, specs_dir):
    files = []
    audio_data = {}
    for file in os.listdir(data_dir)[1:]:
        new_dir = data_dir + file + "/"
        for wav_file in os.listdir(new_dir):
            if wav_file.endswith(".wav"):
                audio_path = os.path.join(new_dir, wav_file)
                y, sr = librosa.load(audio_path, sr=None)
                audio_data[wav_file] = (y, sr)

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
specs_dir = "./mel_spec/"

if __name__ == "__main__":
    generating_mel_specs(data_dir, specs_dir)