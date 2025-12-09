import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from .transforms import AudioAugmentation, FeatureExtractor, SpecAugment

class Dataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, augment=False, noise_paths=None):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.augment = augment

        self.audio_aug = AudioAugmentation(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate, n_mfcc=20)
        self.spec_aug = SpecAugment(p_augment=0.5)

        if augment:
            self.audio_aug.train()
            self.spec_aug.train()
            if noise_paths:
                for n in noise_paths:
                    if os.path.exists(n):
                        self.audio_aug.add_noise_to_library(n)
        else:
            self.audio_aug.eval()
            self.spec_aug.eval()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.augment:
            waveform = self.audio_aug(waveform)

        target_len = self.sample_rate 
        current_len = waveform.shape[-1]

        if current_len > target_len:
            if self.augment:
                max_start = current_len - target_len
                start_idx = torch.randint(0, max_start, (1,)).item()
                waveform = waveform[:, start_idx : start_idx + target_len]
            else:
                waveform = waveform[:, :target_len]
        elif current_len < target_len:
            padding = target_len - current_len
            waveform = F.pad(waveform, (0, padding))

        mfcc = self.feature_extractor(waveform).squeeze(0)
        if self.augment:
            mfcc = self.spec_aug(mfcc)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return mfcc, label