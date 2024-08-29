import os
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
import torchaudio.functional as A
import random
import torchaudio
from torchaudio.utils import download_asset
from torchaudio.io import AudioEffector
import librosa


class DevDataset(Dataset):
    def __init__(self, sample_rate=16000, metafile = ''):
        self.metafile = metafile
        self.sample_rate = sample_rate
        self.fixed_length = 16000 * 8
        self.dev_data = self.process_asv_metadata()



    def __len__(self):
        return len(self.dev_data)

    def process_asv_metadata(self):
        with open(self.metafile) as fin:
            asv_labels = [[x.strip().split()[0],
                           np.array([1,0]) if x.strip().split()[-1] == 'bonafide' else np.array([0,1])] for x in fin.readlines()]
        return asv_labels

    def __getitem__(self, idx):
        waveform, sr = librosa.load(self.dev_data[idx][0], sr=self.sample_rate)
        waveform = torch.Tensor(waveform)
        if len(waveform) > self.fixed_length:
            waveform = waveform[:self.fixed_length]
        else:
            padding = self.fixed_length - len(waveform)
            waveform = F.pad(torch.Tensor(waveform), (0, padding), "constant")

        Ys = self.dev_data[idx][1]
        Ys = torch.from_numpy(Ys).float()
        return waveform, Ys


class TrainDataset(Dataset):
    def __init__(self, sample_rate=16000, metafile = ''):
        self.metafile = metafile
        self.sample_rate = sample_rate
        self.avg_time = sample_rate * 8 # avg time: 7.957780289880806 seconds
        self.training_data = self.process_asv_metadata()
        self.rir_file = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")

    def __len__(self):
        return len(self.training_data)

    def process_asv_metadata(self):
        with open(self.metafile) as fin:
            asv_labels = [[x.strip().split()[0],
                           np.array([1,0]) if x.strip().split()[-1] == 'bonafide' else np.array([0,1])] for x in fin.readlines()]
        return asv_labels

    def random_cropping(self,audio, crop_size = 16000):
        if len(audio) >= crop_size:
            start = random.randint(0, len(audio) - crop_size)
            return audio[start:start + crop_size]
        else:
            return audio

    def apply_rir(self, waveform):
        rir_raw, sr = torchaudio.load(self.rir_file)
        self.rir = rir_raw[:, int(sr * 1.01): int(sr * 1.3)]
        self.rir = self.rir / torch.linalg.vector_norm(self.rir, ord=2)
        waveform = waveform.unsqueeze(0)
        waveform = torchaudio.functional.fftconvolve(waveform, self.rir)
        return waveform.squeeze(0)

    def apply_noise(self, waveform):
        snr = torch.tensor([25])
        noise_level = 80
        noise = torch.randn_like(waveform) * noise_level
#        print(f'waveform shape: {waveform.shape}, noise shape: {noise.shape}')
        waveform = waveform.unsqueeze(0)
        noise = noise.unsqueeze(0)
        noisy_wav = A.add_noise(waveform, noise, snr)
        return noisy_wav.squeeze(0)




    def __getitem__(self, idx):
        waveform, sr = librosa.load(self.training_data[idx][0], sr=self.sample_rate)
        #waveform = self.random_cropping(waveform)
        waveform = torch.Tensor(waveform)



#        if self.training_data[idx][1] == np.array([1.,0.]):
        if np.array_equal(self.training_data[idx][1], np.array([1., 0.])):
            random_number = random.randint(0, 1)
            if random_number == 0:
                waveform = self.apply_rir(waveform)
            elif random_number == 1:
                waveform = self.apply_noise(waveform)

        if len(waveform) > self.avg_time:
            waveform = waveform[:self.avg_time]
        else:
            padding = self.avg_time - len(waveform)
            waveform = F.pad(torch.Tensor(waveform), (0, padding), "constant")





        Ys = self.training_data[idx][1]
        Ys = torch.from_numpy(Ys).float()
        return waveform, Ys
