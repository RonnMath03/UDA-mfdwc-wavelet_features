import librosa
import pywt
import numpy as np
import torch
import torch.nn as nn

class MFDWCFeatureExtractor(nn.Module):
    """
    Mel-Frequency Discrete Wavelet Cepstral Coefficients (MFDWC) Feature Extractor
    
    Replaces DCT with DWT in traditional MFCC pipeline.
    DWT is applied to log mel-filterbank energies (spectral domain).
    Uses optimal configuration: WA ⊕ ΔA ⊕ WD
    """
    
    def __init__(self, n_mels=60, n_fft=2048, hop_length=256, 
                 wavelet='haar', sample_rate=44100):
        """
        Args:
            n_mels: Number of mel-filterbank bands (default: 60)
            n_fft: FFT size (default: 2048)
            hop_length: Hop length for STFT (default: 256)
            wavelet: Wavelet type for DWT (default: 'haar')
            sample_rate: Audio sample rate (default: 44100)
        """
        super(MFDWCFeatureExtractor, self).__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.wavelet = wavelet
        self.sample_rate = sample_rate
        
        # Pre-emphasis filter coefficient
        self.pre_emphasis = 0.97
        
        # Frame parameters (20ms with 50% overlap)
        self.frame_length = int(0.02 * sample_rate)  # 20ms
        self.frame_hop = self.frame_length // 2      # 50% overlap
        
    def forward(self, waveform):
        """
        Extract MFDWC features from audio waveform
        
        Args:
            waveform: Tensor of shape (batch_size, audio_length)
            
        Returns:
            features: Tensor of shape (batch_size, feature_dim)
                    For n_mels=60: (batch_size, 180)
                    - 90 mean features (30 WA + 30 ΔA + 30 WD)
                    - 90 std features (30 WA + 30 ΔA + 30 WD)
        """
        batch_size = waveform.shape[0]
        all_frame_features = []
        
        for i in range(batch_size):
            audio = waveform[i].cpu().numpy()
            
            # Extract frame-wise MFDWC features
            frame_features = self.extract_mfdwc_frames(audio)
            all_frame_features.append(torch.tensor(frame_features, dtype=torch.float32))
        
        # Stack features: (batch_size, n_frames, n_features)
        stacked_features = torch.stack(all_frame_features).to(waveform.device)
        
        # Compute frame-wise statistics (for CNN input)
        mean_features = torch.mean(stacked_features, dim=1)  # (batch_size, n_features)
        std_features = torch.std(stacked_features, dim=1)    # (batch_size, n_features)
        
        # Concatenate mean and std as final feature vector
        final_features = torch.cat([mean_features, std_features], dim=1)
        
        return final_features
    
    def extract_mfdwc_frames(self, audio):
        """
        Extract MFDWC features frame by frame
        
        DWT is applied to log mel-filterbank energies (spectral domain),
        NOT to temporal sequences within mel bands.
        
        Args:
            audio: 1D numpy array of audio samples
            
        Returns:
            features: 2D array of shape (n_frames, n_features)
                     where n_features = 1.5 * n_mfdwc (WA + ΔA + WD)
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
        
        # Process each frame
        for frame in windowed_frames.T:
            # 1. Compute STFT
            fft_frame = np.fft.rfft(frame, n=self.n_fft)
            magnitude = np.abs(fft_frame)
            power_spectrum = magnitude ** 2
            
            # 2. Apply mel filterbank
            mel_filters = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmax=self.sample_rate / 2
            )
            
            # 3. Get mel energies and take logarithm
            mel_energies = np.dot(mel_filters, power_spectrum)
            log_mel_energies = np.log(mel_energies + 1e-10)
            
            # 4. KEY STEP: Apply DWT instead of DCT
            # This is where MFDWC differs from MFCC
            approx_coeffs, detail_coeffs = self.apply_dwt_to_mel_energies(log_mel_energies)
            
            all_approx_coeffs.append(approx_coeffs)
            all_detail_coeffs.append(detail_coeffs)
        
        # Convert to numpy arrays
        approx_features = np.array(all_approx_coeffs)  # (n_frames, n_approx_coeffs)
        detail_features = np.array(all_detail_coeffs)  # (n_frames, n_detail_coeffs)
        
        # Compute delta features for approximation coefficients (across time)
        delta_approx = self.compute_delta_features(approx_features)
        
        # Optimal configuration: WA ⊕ ΔA ⊕ WD
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
        
        Paper methodology:
        - 60 mel bands → 1-level DWT → ~30 approx + ~30 detail coefficients
        - ALL coefficients are kept (no truncation)
        
        Args:
            log_mel_energies: 1D array of log mel-filterbank energies for one frame
                            Shape: (n_mels,) e.g., (60,)
            
        Returns:
            approx_coeffs: Approximation coefficients (~30 for 60 mel bands)
            detail_coeffs: Detail coefficients (~30 for 60 mel bands)
        """
        # Apply 1-level DWT to the log mel energies vector
        coeffs = pywt.dwt(log_mel_energies, self.wavelet)
        approx_coeffs, detail_coeffs = coeffs
        
        # Paper keeps ALL coefficients from DWT decomposition
        # For 60 mel bands with Haar wavelet: approx ≈ 30, detail ≈ 30
        return approx_coeffs, detail_coeffs
    
    def compute_delta_features(self, features, window=3):
        """
        Compute delta (velocity) features across time dimension
        
        Args:
            features: 2D array of shape (n_frames, n_coeffs)
            window: Window size for delta computation (default: 3)
            
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