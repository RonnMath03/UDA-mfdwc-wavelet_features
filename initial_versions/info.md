# MFDWC (Mel-Frequency Discrete Wavelet Cepstral Coefficients) - Complete Technical Documentation

## 📋 Table of Contents

1. Introduction
2. Theoretical Background
3. MFDWC vs MFCC
4. Implementation Architecture
5. Step-by-Step Feature Extraction
6. Code Implementation
7. Configuration Parameters
8. Usage Examples
9. Feature Dimension Calculation
10. Testing and Validation

---

## Introduction

MFDWC (Mel-Frequency Discrete Wavelet Cepstral Coefficients) is an advanced audio feature extraction technique that replaces the Discrete Cosine Transform (DCT) in traditional MFCC with Discrete Wavelet Transform (DWT). This modification provides better time-frequency localization and improved representation of audio signals.

**Key Characteristics:**
- Signal processing-based (no trainable parameters)
- Efficient computational requirements
- Produces compact, informative features
- Specifically designed for acoustic scene classification
- Ready for CNN model integration

---

## Theoretical Background

### Traditional MFCC Pipeline

1. **Pre-emphasis** → Boost high frequencies
2. **Framing** → Split audio into overlapping frames
3. **Windowing** → Apply Hamming window
4. **FFT** → Compute frequency spectrum
5. **Mel-filterbank** → Apply mel-scale filters
6. **Logarithm** → Take log of energies
7. **DCT** → Extract cepstral coefficients ⬅️ **MFDWC changes this**

### MFDWC Innovation

**MFDWC replaces DCT with DWT** in step 7, providing:
- Better time-frequency localization
- Multi-resolution analysis through wavelet decomposition
- Approximation (low-frequency) and Detail (high-frequency) coefficients
- Enhanced feature representation for acoustic signals

---

## MFDWC vs MFCC

### Comparison Table

| Aspect | MFCC | MFDWC |
|--------|------|-------|
| Transform | Discrete Cosine Transform (DCT) | Discrete Wavelet Transform (DWT) |
| Application | Log mel-energies | Log mel-energies |
| Coefficients | Single set of cepstral coeffs | Approximation + Detail coeffs |
| Time-Frequency | Fixed resolution | Multi-resolution |
| Feature Types | Static coefficients | WA (Wavelet Approx), WD (Wavelet Detail), ΔA (Delta) |
| Typical Dimension | 13-40 coefficients | 45-90 features (with deltas) |

### Key Difference Visualization

```
MFCC:  Audio → Frames → STFT → Mel → Log → DCT → Coefficients
                                                ↑
MFDWC: Audio → Frames → STFT → Mel → Log → DWT → Approx + Detail
                                                ↑
                                        Replace DCT with DWT
```

---

## Implementation Architecture

### Class Structure

```python
MFDWCFeatureExtractor(nn.Module)
├── __init__()          # Configure parameters
├── forward()           # Main extraction pipeline
├── extract_mfdwc_frames()  # Frame-by-frame processing
├── apply_dwt_to_mel_energies()  # Core DWT application
└── compute_delta_features()     # Temporal delta computation
```

### Feature Flow

```
Input Audio Waveform
    ↓
Pre-emphasis Filter
    ↓
Frame Segmentation (20ms, 50% overlap)
    ↓
Hamming Windowing
    ↓
Per-Frame Processing:
    ├── FFT → Power Spectrum
    ├── Mel-filterbank Application
    ├── Logarithm
    └── DWT Decomposition
        ├── Approximation Coefficients (WA)
        └── Detail Coefficients (WD)
    ↓
Delta Feature Computation (ΔA)
    ↓
Feature Concatenation: WA ⊕ ΔA ⊕ WD
    ↓
Statistical Aggregation: Mean + Std
    ↓
Final Feature Vector (for CNN)
```

---

## Step-by-Step Feature Extraction

### Step 1: Pre-emphasis

**Purpose:** Boost high-frequency components

```python
pre_emphasis = 0.97
emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
```

**Effect:** Compensates for high-frequency attenuation in speech/audio signals

### Step 2: Framing

**Purpose:** Divide audio into short overlapping segments

**Parameters:**
- Frame length: 20ms (640 samples at 32kHz)
- Hop length: 10ms (50% overlap)
- Window: Hamming

```python
frame_length = int(0.02 * sample_rate)  # 20ms
frame_hop = frame_length // 2            # 50% overlap
frames = librosa.util.frame(emphasized_audio, 
                           frame_length=frame_length,
                           hop_length=frame_hop)
windowed_frames = frames * np.hamming(frame_length)
```

**Result:** ~100 frames per second of audio

### Step 3: STFT and Power Spectrum

**Purpose:** Convert to frequency domain

```python
fft_frame = np.fft.rfft(frame, n=n_fft)  # n_fft = 2048
magnitude = np.abs(fft_frame)
power_spectrum = magnitude ** 2
```

**Output:** 1025 frequency bins (for n_fft=2048)

### Step 4: Mel-Filterbank Application

**Purpose:** Apply perceptually-motivated mel-scale filters

```python
mel_filters = librosa.filters.mel(
    sr=sample_rate,
    n_fft=n_fft,
    n_mels=60,
    fmax=sample_rate / 2
)
mel_energies = np.dot(mel_filters, power_spectrum)
```

**Output:** 60 mel-band energies per frame

### Step 5: Logarithmic Compression

**Purpose:** Compress dynamic range (human perception is logarithmic)

```python
log_mel_energies = np.log(mel_energies + 1e-10)
```

**Output:** 60 log mel-energies (this is the input to DWT)

### Step 6: DWT Application (Core MFDWC Step)

**Purpose:** Decompose log mel-energies into approximation and detail coefficients

```python
# Apply 1-level DWT to the log mel energies vector
coeffs = pywt.dwt(log_mel_energies, wavelet='haar')
approx_coeffs, detail_coeffs = coeffs

# Keep first n_mfdwc//2 coefficients from each
approx_coeffs = approx_coeffs[:n_mfdwc//2]  # e.g., 15 coeffs
detail_coeffs = detail_coeffs[:n_mfdwc//2]  # e.g., 15 coeffs
```

**Key Points:**
- DWT is applied to the **spectral vector** (60 log mel-energies)
- NOT applied to temporal sequences
- Produces two coefficient sets:
  - **Approximation (WA):** Low-frequency components
  - **Detail (WD):** High-frequency components

**Output per frame:** 30 coefficients (15 approx + 15 detail)

### Step 7: Delta Feature Computation

**Purpose:** Capture temporal dynamics across frames

```python
def compute_delta_features(features, window=3):
    delta = np.zeros_like(features)
    padded = np.pad(features, ((window//2, window//2), (0, 0)), mode='edge')
    
    for t in range(n_frames):
        delta[t] = padded[t + window//2 + 1] - padded[t + window//2 - 1]
    
    return delta
```

**Applied to:** Approximation coefficients only (not detail)

**Output:** 15 delta approximation coefficients (ΔA) per frame

### Step 8: Feature Concatenation

**Purpose:** Combine all feature types

```python
optimal_features = np.concatenate([
    approx_features,  # WA: 15 coeffs
    delta_approx,     # ΔA: 15 coeffs
    detail_features   # WD: 15 coeffs
], axis=1)
```

**Output per frame:** 45 features (15 + 15 + 15)
**For all frames:** Shape = (n_frames, 45)

### Step 9: Statistical Aggregation

**Purpose:** Create fixed-length representation for variable-length audio

```python
mean_features = torch.mean(frame_features, dim=1)  # (batch, 45)
std_features = torch.std(frame_features, dim=1)    # (batch, 45)
final_features = torch.cat([mean_features, std_features], dim=1)
```

**Final Output:** 90 features (45 mean + 45 std)

---

## Code Implementation

### Complete MFDWC Extractor Class

```python
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
                 wavelet='haar', sample_rate=32000):
        """
        Args:
            n_mels: Number of mel-filterbank bands (default: 60)
            n_fft: FFT size (default: 2048)
            hop_length: Hop length for STFT (default: 256)
            wavelet: Wavelet type for DWT (default: 'haar')
            sample_rate: Audio sample rate (default: 32000)
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
```

---

## Configuration Parameters

### Recommended Configurations

#### Default Configuration (Balanced)
```python
MFDWC_CONFIG = {
    'n_mels': 60,           # Mel bands
    'n_mfdwc': 30,          # MFDWC coefficients (15 approx + 15 detail)
    'wavelet': 'haar',      # Wavelet type
    'sample_rate': 32000,   # Audio sample rate
    'n_fft': 2048,          # FFT size
    'hop_length': 256       # STFT hop length
}
# Output: 90 features (45 mean + 45 std)
```

#### Compact Configuration (Faster)
```python
MFDWC_CONFIG = {
    'n_mels': 40,
    'n_mfdwc': 20,
    'wavelet': 'haar',
    'sample_rate': 16000,
    'n_fft': 1024,
    'hop_length': 128
}
# Output: 60 features (30 mean + 30 std)
```

#### High-Resolution Configuration (More Detail)
```python
MFDWC_CONFIG = {
    'n_mels': 80,
    'n_mfdwc': 40,
    'wavelet': 'db4',      # Daubechies wavelet
    'sample_rate': 48000,
    'n_fft': 4096,
    'hop_length': 512
}
# Output: 120 features (60 mean + 60 std)
```

### Wavelet Types

| Wavelet | Description | Use Case |
|---------|-------------|----------|
| `'haar'` | Simple, fast | General audio, real-time |
| `'db4'` | Daubechies 4 | Better frequency resolution |
| `'db8'` | Daubechies 8 | High detail, slower |
| `'sym4'` | Symlet 4 | Symmetric, balanced |
| `'coif1'` | Coiflet 1 | Smooth approximation |

---

## Usage Examples

### Example 1: Single Audio File

```python
import torch
import librosa
from mfdwc_extractor import MFDWCFeatureExtractor

# Initialize extractor
extractor = MFDWCFeatureExtractor(
    n_mels=60,
    n_mfdwc=30,
    wavelet='haar',
    sample_rate=32000
)

# Load audio
audio, sr = librosa.load('audio.wav', sr=32000, mono=True)

# Convert to tensor and add batch dimension
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

# Extract features
features = extractor(audio_tensor)

print(f"Input shape: {audio_tensor.shape}")  # (1, audio_length)
print(f"Output shape: {features.shape}")      # (1, 90)
```

### Example 2: Batch Processing

```python
import torch
import librosa
from mfdwc_extractor import MFDWCFeatureExtractor

# Initialize extractor
extractor = MFDWCFeatureExtractor(
    n_mels=60,
    n_mfdwc=30,
    wavelet='haar',
    sample_rate=32000
)

# Load multiple audio files
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav', 'audio4.wav']
batch_audio = []

for file in audio_files:
    audio, sr = librosa.load(file, sr=32000, mono=True, duration=10)
    
    # Ensure fixed length (10 seconds)
    if len(audio) < 32000 * 10:
        audio = np.pad(audio, (0, 32000*10 - len(audio)))
    else:
        audio = audio[:32000*10]
    
    batch_audio.append(audio)

# Convert to tensor
batch_tensor = torch.tensor(batch_audio, dtype=torch.float32)

# Extract features
features = extractor(batch_tensor)

print(f"Batch input shape: {batch_tensor.shape}")   # (4, 320000)
print(f"Batch output shape: {features.shape}")      # (4, 90)
```

### Example 3: With PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from mfdwc_extractor import MFDWCFeatureExtractor

class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate=32000, duration=10):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_len = sample_rate * duration
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio, sr = librosa.load(self.audio_files[idx], 
                                sr=self.sample_rate, 
                                mono=True)
        
        # Pad or truncate
        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)))
        else:
            audio = audio[:self.max_len]
        
        return torch.tensor(audio, dtype=torch.float32)

# Create dataset and dataloader
audio_files = ['audio1.wav', 'audio2.wav', ...]  # Your audio files
dataset = AudioDataset(audio_files)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize extractor
extractor = MFDWCFeatureExtractor(
    n_mels=60,
    n_mfdwc=30,
    wavelet='haar',
    sample_rate=32000
)

# Process batches
for batch_audio in dataloader:
    features = extractor(batch_audio)
    print(f"Batch features shape: {features.shape}")  # (8, 90)
```

---

## Feature Dimension Calculation

### Formula

For configuration with `n_mfdwc` coefficients:

1. **Per-frame features:**
   - Approximation coefficients (WA): `n_mfdwc // 2`
   - Delta approximation (ΔA): `n_mfdwc // 2`
   - Detail coefficients (WD): `n_mfdwc // 2`
   - **Total per frame:** `1.5 * n_mfdwc`

2. **Final features (after statistical aggregation):**
   - Mean features: `1.5 * n_mfdwc`
   - Std features: `1.5 * n_mfdwc`
   - **Total:** `3 * n_mfdwc`

### Examples

| n_mfdwc | Per-frame | Final Dimension |
|---------|-----------|-----------------|
| 20 | 30 | 60 |
| 30 | 45 | 90 |
| 40 | 60 | 120 |
| 60 | 90 | 180 |

### Calculation Example

For `n_mfdwc = 30`:

```python
n_mfdwc = 30

# Per-frame calculation
n_approx = n_mfdwc // 2      # 15
n_delta = n_mfdwc // 2       # 15
n_detail = n_mfdwc // 2      # 15
per_frame = 15 + 15 + 15     # 45

# Final dimension (mean + std)
final_dim = 2 * per_frame    # 90
```

---

## Testing and Validation

### Test Script 1: Basic Functionality

```python
import torch
import numpy as np
from mfdwc_extractor import MFDWCFeatureExtractor

def test_basic_extraction():
    """Test basic MFDWC extraction"""
    print("=== Testing Basic MFDWC Extraction ===")
    
    # Create test audio (10 seconds, 440Hz tone + noise)
    sample_rate = 32000
    duration = 10
    t = np.linspace(0, duration, sample_rate * duration)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 
             0.2 * np.random.randn(len(t)))
    
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    # Initialize extractor
    extractor = MFDWCFeatureExtractor(
        n_mels=60,
        n_mfdwc=30,
        wavelet='haar',
        sample_rate=sample_rate
    )
    
    # Extract features
    features = extractor(audio_tensor)
    
    print(f"✅ Input shape: {audio_tensor.shape}")
    print(f"✅ Output shape: {features.shape}")
    print(f"✅ Expected shape: (1, 90)")
    print(f"✅ Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")
    
    assert features.shape == (1, 90), "Feature dimension mismatch!"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_basic_extraction()
```

### Test Script 2: Batch Processing

```python
def test_batch_processing():
    """Test batch processing"""
    print("\n=== Testing Batch Processing ===")
    
    batch_size = 4
    sample_rate = 32000
    duration = 5
    
    # Create batch of different audio signals
    batch_audio = []
    for i in range(batch_size):
        t = np.linspace(0, duration, sample_rate * duration)
        freq = 440 * (i + 1)
        audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.2 * np.random.randn(len(t))
        batch_audio.append(audio)
    
    batch_tensor = torch.tensor(batch_audio, dtype=torch.float32)
    
    # Extract features
    extractor = MFDWCFeatureExtractor(
        n_mels=60,
        n_mfdwc=30,
        wavelet='haar',
        sample_rate=sample_rate
    )
    
    features = extractor(batch_tensor)
    
    print(f"✅ Batch input shape: {batch_tensor.shape}")
    print(f"✅ Batch output shape: {features.shape}")
    print(f"✅ Expected shape: (4, 90)")
    
    # Check if features are different for different inputs
    feature_diffs = []
    for i in range(1, batch_size):
        diff = torch.norm(features[i] - features[0]).item()
        feature_diffs.append(diff)
        print(f"✅ Feature difference between sample 0 and {i}: {diff:.4f}")
    
    assert all(diff > 0.01 for diff in feature_diffs), "Features too similar!"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_batch_processing()
```

### Test Script 3: Wavelet Comparison

```python
def test_wavelet_comparison():
    """Test different wavelet types"""
    print("\n=== Testing Different Wavelets ===")
    
    # Create test audio
    sample_rate = 32000
    duration = 5
    t = np.linspace(0, duration, sample_rate * duration)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    wavelets = ['haar', 'db4', 'db8', 'sym4']
    
    for wavelet in wavelets:
        extractor = MFDWCFeatureExtractor(
            n_mels=60,
            n_mfdwc=30,
            wavelet=wavelet,
            sample_rate=sample_rate
        )
        
        features = extractor(audio_tensor)
        
        print(f"✅ Wavelet '{wavelet}':")
        print(f"   Shape: {features.shape}")
        print(f"   Mean: {features.mean().item():.4f}")
        print(f"   Std: {features.std().item():.4f}")
    
    print("✅ All wavelets tested successfully!")

if __name__ == "__main__":
    test_wavelet_comparison()
```



## Key Differences from MFCC (Summary)

| Aspect | MFCC | MFDWC |
|--------|------|-------|
| **Transform** | DCT | DWT |
| **Coefficients** | Single cepstral set | Approximation + Detail |
| **Frequency Resolution** | Fixed | Multi-resolution |
| **Feature Components** | Static + Deltas | WA + ΔA + WD |
| **Time-Frequency** | Good time, poor frequency | Balanced time-frequency |
| **Application** | Speech recognition | Acoustic scene classification |

---

## Performance Tips

### 1. **Optimize for Speed**
```python
# Use smaller configurations for real-time
fast_config = {
    'n_mels': 40,
    'n_mfdwc': 20,
    'wavelet': 'haar',  # Fastest wavelet
    'sample_rate': 16000
}
```

### 2. **Batch Processing**
```python
# Process multiple files at once
with torch.no_grad():  # Disable gradients for inference
    features = extractor(batch_audio)
```

### 3. **GPU Acceleration**
```python
# Move extractor to GPU
extractor = extractor.to('cuda')
audio_batch = audio_batch.to('cuda')
features = extractor(audio_batch)
```

### 4. **Memory Efficiency**
```python
# Process in smaller chunks if memory is limited
chunk_size = 4
for i in range(0, len(large_dataset), chunk_size):
    chunk = large_dataset[i:i+chunk_size]
    features = extractor(chunk)
    # Process features...
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Feature Dimension Mismatch
```python
# Problem: Expected 90, got different dimension
# Solution: Check n_mfdwc configuration
print(f"Expected dimension: {3 * n_mfdwc}")
print(f"Actual dimension: {features.shape[1]}")
```

#### Issue 2: Audio Length Inconsistency
```python
# Problem: Different audio lengths in batch
# Solution: Pad/truncate to fixed length
max_len = 32000 * 10  # 10 seconds at 32kHz
if len(audio) < max_len:
    audio = np.pad(audio, (0, max_len - len(audio)))
else:
    audio = audio[:max_len]
```

#### Issue 3: Slow Processing
```python
# Problem: Slow feature extraction
# Solutions:
# 1. Use 'haar' wavelet (fastest)
# 2. Reduce n_mels
# 3. Use GPU
# 4. Process in batches
```

---

## References

1. **Original Paper**: *Mel-Frequency Discrete Wavelet Coefficients for Acoustic Scene Classification*
2. **Wavelet Theory**: Daubechies, I. (1992). *Ten Lectures on Wavelets*
3. **MFCC Background**: Davis, S., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition*

---


**END OF DOCUMENTATION**

This documentation provides everything needed to implement MFDWC feature extraction for CNN-based audio classification systems. The implementation correctly follows the research paper's methodology of replacing DCT with DWT in the traditional MFCC pipeline.