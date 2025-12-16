import torch
import numpy as np
from mfdwc_extractor import MFDWCFeatureExtractor
import matplotlib.pyplot as plt

def debug_mfdwc_step_by_step():
    """Debug MFDWC extraction step by step"""
    print("=== Debugging MFDWC Extraction Step by Step ===")
    
    # Create simple test audio
    sample_rate = 32000
    duration = 2  # Start with shorter duration for debugging
    t = np.linspace(0, duration, sample_rate * duration)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))
    
    print(f"Input audio length: {len(audio)} samples ({duration}s)")
    
    # Initialize extractor with smaller parameters for debugging
    extractor = MFDWCFeatureExtractor(
        n_mels=20,  # Reduced for debugging
        n_mfdwc=20,  # Reduced for debugging
        wavelet='haar',
        sample_rate=sample_rate
    )
    
    # Test step by step
    try:
        # Step 1: Pre-emphasis
        audio_pre = np.append(audio[0], audio[1:] - extractor.pre_emphasis * audio[:-1])
        print(f"✅ Pre-emphasis: {len(audio_pre)} samples")
        
        # Step 2: Framing
        frame_length = int(0.02 * sample_rate)  # 20ms
        hop_length_frames = frame_length // 2
        print(f"Frame length: {frame_length}, hop length: {hop_length_frames}")
        
        import librosa
        frames = librosa.util.frame(audio_pre, frame_length=frame_length, 
                                  hop_length=hop_length_frames, axis=0)
        frames = frames * np.hamming(frame_length)
        print(f"✅ Framing: {frames.shape}")
        
        # Step 3: Mel features for first few frames only
        mel_features = []
        for i, frame in enumerate(frames.T[:5]):  # Test first 5 frames
            fft = np.fft.rfft(frame, n=extractor.n_fft)
            magnitude = np.abs(fft)
            power = magnitude ** 2
            
            mel_filters = librosa.filters.mel(
                sr=sample_rate, 
                n_fft=extractor.n_fft, 
                n_mels=extractor.n_mels,
                fmax=sample_rate/2
            )
            
            mel_energy = np.dot(mel_filters, power[:len(mel_filters[0])])
            log_mel_energy = np.log(mel_energy + 1e-10)
            mel_features.append(log_mel_energy)
        
        mel_features = np.array(mel_features).T
        print(f"✅ Mel features: {mel_features.shape}")
        
        # Step 4: DWT for first mel band
        import pywt
        test_band = mel_features[0]  # First mel band
        coeffs = pywt.dwt(test_band, extractor.wavelet)
        approx_coeffs, detail_coeffs = coeffs
        print(f"✅ DWT coefficients - Approx: {len(approx_coeffs)}, Detail: {len(detail_coeffs)}")
        
        # Step 5: Test delta computation with small array
        test_features = np.random.randn(5, 10)  # 5 mels, 10 coeffs
        delta_test = extractor.compute_delta_features(test_features)
        print(f"✅ Delta features: input {test_features.shape} -> output {delta_test.shape}")
        
        print("\n🎉 All individual steps work! Now testing full extraction...")
        
        # Full extraction test
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        features = extractor(audio_tensor)
        print(f"✅ Full extraction successful: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_configurations():
    """Test MFDWC with different configurations"""
    print("\n=== Testing Different Configurations ===")
    
    # Create test audio
    sample_rate = 32000
    duration = 5
    t = np.linspace(0, duration, sample_rate * duration)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    configs = [
        {"n_mels": 30, "n_mfdwc": 30, "desc": "Small config"},
        {"n_mels": 60, "n_mfdwc": 60, "desc": "Original config"},
        {"n_mels": 40, "n_mfdwc": 40, "desc": "Medium config"},
    ]
    
    for config in configs:
        try:
            extractor = MFDWCFeatureExtractor(
                n_mels=config["n_mels"],
                n_mfdwc=config["n_mfdwc"],
                wavelet='haar',
                sample_rate=sample_rate
            )
            
            features = extractor(audio_tensor)
            expected_dim = config["n_mels"] * (config["n_mfdwc"] + config["n_mfdwc"]//2)
            
            print(f"✅ {config['desc']}: {features.shape} (expected ~{expected_dim})")
            
        except Exception as e:
            print(f"❌ {config['desc']} failed: {e}")

if __name__ == "__main__":
    print("🔍 MFDWC Debug Testing\n")
    
    # Step 1: Debug step by step
    if debug_mfdwc_step_by_step():
        # Step 2: Test different configurations
        test_with_different_configurations()
        print("\n🎉 All debug tests completed!")
    else:
        print("\n❌ Debug tests failed. Check the implementation.")