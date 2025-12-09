import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

class AudioAugmentation(nn.Module):
    def __init__(self, 
                 p_white_noise=0.4, 
                 p_background_noise=0.6, 
                 p_pitch_shift=0.0,
                 p_time_stretch=0.0,
                 sample_rate=16000):
        super().__init__()

        self.sample_rate = sample_rate
        self.p_white_noise = p_white_noise
        self.p_background_noise = p_background_noise
        self.p_pitch_shift = p_pitch_shift
        self.p_time_stretch = p_time_stretch

        self.noise_library = []

        self.pitch_shifter_up = torchaudio.transforms.PitchShift(sample_rate, n_steps=2)
        self.pitch_shifter_down = torchaudio.transforms.PitchShift(sample_rate, n_steps=-2)
    
    def add_noise_to_library(self, noise_path):
        waveform, sr = torchaudio.load(noise_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        self.noise_library.append(waveform)
    
    def apply_white_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def apply_background_noise(self, waveform, min_snr=8, max_snr=20):
        if not self.noise_library:
            return waveform
        noise = self.noise_library[torch.randint(0, len(self.noise_library), (1,)).item()]
        noise = noise.to(waveform.device)

        audio_len = waveform.shape[-1]
        noise_len = noise.shape[-1]
        
        if noise_len < audio_len:
            repeats = (audio_len // noise_len) + 1
            noise = noise.repeat(1, repeats)
            noise = noise[..., :audio_len] 
        elif noise_len > audio_len:
            start = torch.randint(0, noise_len - audio_len, (1,)).item()
            noise = noise[..., start:start + audio_len]
        
        snr = torch.tensor([torch.randint(min_snr, max_snr, (1,)).item()], device=waveform.device)
        
        return F.add_noise(waveform, noise, snr)
    
    def apply_time_stretch(self, waveform, min_rate=0.8, max_rate=1.2):
        rate = min_rate + (max_rate - min_rate) * torch.rand(1).item()
        
        if rate == 1.0:
            return waveform

        new_sample_rate = int(self.sample_rate * rate)
        
        resampler = T.Resample(
            orig_freq=new_sample_rate, 
            new_freq=self.sample_rate
        ).to(waveform.device) 
        
        stretched_waveform = resampler(waveform)
        
        return stretched_waveform
    
    def forward(self, waveform):

        if not self.training:
            return waveform

        if torch.rand(1).item() < self.p_time_stretch:
            waveform = self.apply_time_stretch(waveform)
            
        if torch.rand(1).item() < self.p_pitch_shift:
            if torch.rand(1).item() < 0.5:
                waveform = self.pitch_shifter_up(waveform)
            else:
                waveform = self.pitch_shifter_down(waveform)

        if torch.rand(1).item() < self.p_white_noise:
            random_level = torch.rand(1).item() * 0.005 + 0.001
            waveform = self.apply_white_noise(waveform, noise_level=random_level)

        if torch.rand(1).item() < self.p_background_noise:
            waveform = self.apply_background_noise(waveform)
            
        return waveform
    
class SpecAugment(nn.Module):
    def __init__(self, p_augment=0.4):
        super().__init__()
        self.p_augment = p_augment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
        self.time_mask = T.TimeMasking(time_mask_param=20)

    def forward(self, features):
        if torch.rand(1).item() < self.p_augment:
            features = self.freq_mask(features)
            features = self.time_mask(features)
        return features
    
class FeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=20):
        super().__init__()
        self.mfcc = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, 
            melkwargs={
                "n_fft": 512,
                "n_mels": 40,
                "hop_length": 160,
                "mel_scale": "htk",
            })
        
    def forward(self, waveform):
        return self.mfcc(waveform)
