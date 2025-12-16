import librosa
import pywt
import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct

class MFDWCFeatureExtractor(nn.Module):
    """
    Mel-Frequency Discrete Wavelet Cepstral Coefficients (MFDWC) Feature Extractor
    
    Implements the correct methodology as per the paper:
    - DWT replaces DCT in the traditional MFCC pipeline
    - Applied to log mel-filterbank energies (spectral domain), not temporal sequences
    - Uses optimal configuration: WA ⊕ ΔA ⊕ WD
    """
    
    def __init__(self, n_mels=60, n_fft=2048, hop_length=256, 
                 wavelet='haar', n_mfdwc=30, sample_rate=32000):
        super(MFDWCFeatureExtractor, self).__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.wavelet = wavelet
        self.n_mfdwc = n_mfdwc
        self.sample_rate = sample_rate
        
        # Pre-emphasis filter coefficient
        self.pre_emphasis = 0.97
        
        # Frame parameters (20ms with 50% overlap as per paper)
        self.frame_length = int(0.02 * sample_rate)  # 20ms
        self.frame_hop = self.frame_length // 2      # 50% overlap
        
    def forward(self, waveform):
        """
        Extract MFDWC features from audio waveform
        
        Args:
            waveform: Tensor of shape (batch_size, audio_length)
            
        Returns:
            features: Tensor of shape (batch_size, feature_dim)
                     where feature_dim = 2 * n_mfdwc_total (mean + std)
        """
        batch_size = waveform.shape[0]
        all_frame_features = []
        
        for i in range(batch_size):
            audio = waveform[i].cpu().numpy()
            
            # Extract frame-wise MFDWC features following paper's methodology
            frame_features = self.extract_mfdwc_frames(audio)
            all_frame_features.append(torch.tensor(frame_features, dtype=torch.float32))
        
        # Stack features: (batch_size, n_frames, n_features)
        stacked_features = torch.stack(all_frame_features).to(waveform.device)
        
        # Compute frame-wise statistics as per paper (for SVM input)
        mean_features = torch.mean(stacked_features, dim=1)  # (batch_size, n_features)
        std_features = torch.std(stacked_features, dim=1)    # (batch_size, n_features)
        
        # Concatenate mean and std as final feature vector
        final_features = torch.cat([mean_features, std_features], dim=1)
        
        return final_features
    
    def extract_mfdwc_frames(self, audio):
        """
        Extract MFDWC features frame by frame following the paper's methodology
        
        Key difference from incorrect implementation:
        - DWT is applied to log mel-filterbank energies (spectral domain)
        - NOT applied to temporal sequences within mel bands
        """
        # Pre-emphasis
        emphasized_audio = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        
        # Frame the audio with Hamming window
        frames = librosa.util.frame(
            emphasized_audio, 
            frame_length=self.frame_length,
            hop_length=self.frame_hop,
            axis=0
        )
        
        # Apply Hamming window
        windowed_frames = frames * np.hamming(self.frame_length)
        
        all_approx_coeffs = []
        all_detail_coeffs = []
        
        # Process each frame - this is where MFDWC differs from the old implementation
        for frame in windowed_frames.T:
            # 1. Compute STFT (same as MFCC)
            fft_frame = np.fft.rfft(frame, n=self.n_fft)
            magnitude = np.abs(fft_frame)
            power_spectrum = magnitude ** 2
            
            # 2. Apply mel filterbank (same as MFCC)
            mel_filters = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmax=self.sample_rate / 2
            )
            
            # 3. Get mel energies and take logarithm (same as MFCC)
            mel_energies = np.dot(mel_filters, power_spectrum)
            log_mel_energies = np.log(mel_energies + 1e-10)
            
            # 4. KEY DIFFERENCE: Apply DWT instead of DCT
            # In traditional MFCC, DCT would be applied here
            # MFDWC applies DWT to the log mel energies vector
            approx_coeffs, detail_coeffs = self.apply_dwt_to_mel_energies(log_mel_energies)
            
            all_approx_coeffs.append(approx_coeffs)
            all_detail_coeffs.append(detail_coeffs)
        
        # Convert to numpy arrays
        approx_features = np.array(all_approx_coeffs)  # (n_frames, n_approx_coeffs)
        detail_features = np.array(all_detail_coeffs)  # (n_frames, n_detail_coeffs)
        
        # Compute delta features for approximation coefficients only (across time)
        delta_approx = self.compute_delta_features(approx_features)
        
        # Paper's optimal configuration: WA ⊕ ΔA ⊕ WD
        optimal_features = np.concatenate([
            approx_features,  # WA (Wavelet Approximation)
            delta_approx,     # ΔA (Delta of Approximation)  
            detail_features   # WD (Wavelet Detail)
        ], axis=1)
        
        return optimal_features
    
    def apply_dwt_to_mel_energies(self, log_mel_energies):
        """
        Apply DWT to log mel energies - the core of MFDWC
        
        This replaces DCT in traditional MFCC pipeline.
        DWT is applied to the spectral vector (log mel energies), 
        NOT to temporal sequences.
        
        Args:
            log_mel_energies: 1D array of log mel-filterbank energies for one frame
            
        Returns:
            approx_coeffs: Approximation coefficients
            detail_coeffs: Detail coefficients
        """
        # Apply 1-level DWT to the log mel energies vector
        coeffs = pywt.dwt(log_mel_energies, self.wavelet)
        approx_coeffs, detail_coeffs = coeffs
        
        # Keep specified number of coefficients
        n_approx = min(len(approx_coeffs), self.n_mfdwc // 2)
        n_detail = min(len(detail_coeffs), self.n_mfdwc // 2)
        
        # Truncate to desired number of coefficients
        approx_coeffs = approx_coeffs[:n_approx]
        detail_coeffs = detail_coeffs[:n_detail]
        
        # Pad if necessary to maintain consistent dimensions
        if len(approx_coeffs) < self.n_mfdwc // 2:
            pad_size = self.n_mfdwc // 2 - len(approx_coeffs)
            approx_coeffs = np.pad(approx_coeffs, (0, pad_size), mode='constant')
        
        if len(detail_coeffs) < self.n_mfdwc // 2:
            pad_size = self.n_mfdwc // 2 - len(detail_coeffs)
            detail_coeffs = np.pad(detail_coeffs, (0, pad_size), mode='constant')
        
        return approx_coeffs, detail_coeffs
    
    def compute_delta_features(self, features, window=3):
        """
        Compute delta (velocity) features across time dimension
        
        Args:
            features: 2D array of shape (n_frames, n_coeffs)
            window: Window size for delta computation
            
        Returns:
            delta: Delta features of same shape as input
        """
        n_frames, n_coeffs = features.shape
        delta = np.zeros_like(features)
        
        # Pad the features for boundary frames
        padded = np.pad(features, ((window//2, window//2), (0, 0)), mode='edge')
        
        # Compute delta: f[t+1] - f[t-1] for each frame t
        for t in range(n_frames):
            delta[t] = padded[t + window//2 + 1] - padded[t + window//2 - 1]
        
        return delta


class StatisticalAggregator(nn.Module):
    """
    Statistical aggregator for frame-level features
    Computes mean and standard deviation across time dimension
    """
    def __init__(self):
        super(StatisticalAggregator, self).__init__()
        
    def forward(self, frame_features):
        # frame_features shape: (batch_size, n_frames, n_features)
        # Compute mean and std across time dimension
        mean_features = torch.mean(frame_features, dim=1)  # (batch_size, n_features)
        std_features = torch.std(frame_features, dim=1)    # (batch_size, n_features)
        
        # Concatenate mean and std
        aggregated = torch.cat([mean_features, std_features], dim=1)
        return aggregated