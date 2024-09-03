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
import torchaudio.functional as A
import librosa
import speechbrain as sb
from speechbrain.augment.codec import CodecAugment
class DevDataset(Dataset):
    def __init__(self, sample_rate=16000, metafile = ''):
        self.metafile = metafile
        self.sample_rate = sample_rate
        self.fixed_length = 16000 * 3
        self.dev_data = self.process_asv_metadata()



    def __len__(self):
        return len(self.dev_data)

    def process_asv_metadata(self):
        with open(self.metafile) as fin:
            asv_labels = [[x.strip().split()[0],
                           np.array([1,0]) if x.strip().split()[-1] == 'bonafide' else np.array([0,1])] for x in fin.readlines()]
        return asv_labels

    def __getitem__(self, idx):
#        print(f"Processing file: {self.dev_data[idx][0]}")
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
        self.avg_time = sample_rate * 3 # avg time: 7.957780289880806 seconds
        self.training_data = self.process_asv_metadata()
        self.rir_file = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")

    def __len__(self):
        return len(self.training_data)

    def process_asv_metadata(self):
        with open(self.metafile) as fin:
            asv_labels = [[x.strip().split()[0],
                           np.array([1,0]) if x.strip().split()[-1] == 'bonafide' else np.array([0,1])] for x in fin.readlines()]
        return asv_labels



    def apply_rir(self, waveform):
#        print('rir augm')
        rir_raw, sr = torchaudio.load(self.rir_file)
        self.rir = rir_raw[:, int(sr * 1.01): int(sr * 1.3)]
        self.rir = self.rir / torch.linalg.vector_norm(self.rir, ord=2)
        waveform = waveform.unsqueeze(0)
#        print('waveform shape before rir augm:', waveform.shape)
        rir_waveform = torchaudio.functional.fftconvolve(waveform, self.rir)
        original_length = waveform.shape[-1]
        rir_length = rir_waveform.shape[-1]
        if rir_length > original_length:
            start_idx = (rir_length - original_length) // 2
            rir_waveform = rir_waveform[:, start_idx:start_idx + original_length]
#        print('waveform shape after rir augm:', rir_waveform.shape)
        return rir_waveform.squeeze(0)

    def apply_noise(self, waveform):
#        print('noise augm')
        snr = torch.tensor([25])
        noise_level = 80
        noise = torch.randn_like(waveform) * noise_level
        #print(f'waveform shape: {waveform.shape}, noise shape: {noise.shape}')
        waveform = waveform.unsqueeze(0)
        noise = noise.unsqueeze(0)
        noisy_wav = A.add_noise(waveform, noise, snr)
#        print('waveform shape after noise augm:', noisy_wav.shape)
        return noisy_wav.squeeze(0)

    def random_cropping(self, waveform):
        crop_length = int(3 * 16000)
        total_length = waveform.shape[-1]

        if total_length < crop_length:
            padding = crop_length - waveform.shape[-1]
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            waveform = F.pad(torch.Tensor(waveform), (0, padding), "constant")
            return waveform

        crop_type = random.choice(['start','middle', 'end'])

        if crop_type == 'start':
            start_idx = 0
        elif crop_type == 'end':
            start_idx = total_length - crop_length
        elif crop_type == 'middle':
            start_idx =max(0,(total_length - crop_length) // 2) 

        cropped_waveform = waveform[:, start_idx:start_idx + crop_length]
        
        
        return cropped_waveform


    def apply_codec(self, waveform):
        waveform_input = waveform.unsqueeze(0)
        rand_augm = random.randint(0,2)
        if rand_augm == 0:
#            print('mp3 codec')
            waveform = A.apply_codec(waveform_input, self.sample_rate, format="mp3")
        elif rand_augm == 1:
#            print('vorbis codec')
            waveform = A.apply_codec(waveform_input,self.sample_rate,format = 'vorbis',compression = -1)
        elif rand_augm == 2:
#            print('wav codec')
            waveform = A.apply_codec(waveform_input, self.sample_rate, format = 'wav', encoding = 'ULAW', bits_per_sample = 8)


        if waveform.shape[-1] > waveform_input.shape[-1]:
            start_idx = (waveform.shape[-1] - waveform_input.shape[-1]) // 2
            waveform = waveform[:, start_idx:start_idx + waveform_input.shape[-1]]
        return waveform.squeeze(0)
            

 

    def __getitem__(self, idx):
        waveform, sr = librosa.load(self.training_data[idx][0], sr=self.sample_rate)
        waveform = torch.Tensor(waveform).unsqueeze(0)
        #voice activity detection
        st = waveform.shape[1] - torchaudio.functional.vad(waveform, sample_rate=sr, trigger_level=5).shape[1]
        en = waveform.shape[1] - torchaudio.functional.vad(torch.flip(waveform, [1]), sample_rate=sr, trigger_level=5).shape[1]
        waveform = waveform[:, st:waveform.shape[1] - en]
        #random cropping
#        print('waveform shape after VAD:', waveform.shape)
        waveform =self.random_cropping(waveform)
        waveform = waveform.squeeze(0)
 #       print('waveform shape after random crop:', waveform.shape)

        #random codec augumentation
        random_codec = random.randint(0,4)
        if random_codec == 0:
            waveform = self.apply_codec(waveform)
        #random augumentation
        if np.array_equal(self.training_data[idx][1], np.array([1., 0.])):
            random_number = random.randint(0, 9)
            if random_number == 0:
                waveform = self.apply_rir(waveform)
            elif random_number == 1:
                waveform = self.apply_noise(waveform)
               
                        
                
   

 
        Ys = self.training_data[idx][1]
        Ys = torch.from_numpy(Ys).float()
        return waveform, Ys
