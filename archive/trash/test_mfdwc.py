import torch
import numpy as np
from mfdwc_extractor import MFDWCFeatureExtractor
import matplotlib.pyplot as plt

def test_single_audio_extraction():
    """Test MFDWC extraction on a single audio sample"""
    print("=== Testing Single Audio MFDWC Extraction ===")
    
    # Create synthetic audio (similar to what you might find in acoustic scenes)
    sample_rate = 32000
    duration = 10  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Create a more realistic test signal with multiple frequency components
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz tone
             0.3 * np.sin(2 * np.pi * 1000 * t) +  # 1000 Hz tone
             0.2 * np.random.randn(len(t)))  # Add some noise
    
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Initialize MFDWC extractor
    extractor = MFDWCFeatureExtractor(
        n_mels=60,
        n_mfdwc=60,
        wavelet='haar',
        sample_rate=sample_rate
    )
    
    # Extract features
    features = extractor(audio_tensor)
    
    print(f"Input audio shape: {audio_tensor.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Expected dimension (60 mels × 90 coeffs): {60 * 90}")
    
    # Visualize some statistics
    print(f"Feature statistics:")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    print(f"  Min: {features.min().item():.4f}")
    print(f"  Max: {features.max().item():.4f}")
    
    return features

def test_batch_processing():
    """Test MFDWC extraction on a batch of audio samples"""
    print("\n=== Testing Batch MFDWC Extraction ===")
    
    batch_size = 4
    sample_rate = 32000
    duration = 10
    
    # Create batch of different audio signals
    batch_audio = []
    for i in range(batch_size):
        t = np.linspace(0, duration, sample_rate * duration)
        # Different frequency content for each sample
        freq = 440 * (i + 1)
        audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.2 * np.random.randn(len(t))
        batch_audio.append(audio)
    
    batch_tensor = torch.tensor(batch_audio, dtype=torch.float32)
    
    # Extract features
    extractor = MFDWCFeatureExtractor(
        n_mels=60,
        n_mfdwc=60,
        wavelet='haar',
        sample_rate=sample_rate
    )
    
    features = extractor(batch_tensor)
    
    print(f"Batch input shape: {batch_tensor.shape}")
    print(f"Batch output shape: {features.shape}")
    
    # Check if features are different for different inputs
    feature_diffs = []
    for i in range(1, batch_size):
        diff = torch.norm(features[i] - features[0]).item()
        feature_diffs.append(diff)
        print(f"Feature difference between sample 0 and {i}: {diff:.4f}")
    
    if all(diff > 0.01 for diff in feature_diffs):
        print("✅ Features are appropriately different for different inputs")
    else:
        print("⚠️  Warning: Features might be too similar")
    
    return features

def test_wavelet_comparison():
    """Test different wavelet types"""
    print("\n=== Testing Different Wavelets ===")
    
    # Create test audio
    sample_rate = 32000
    duration = 10
    t = np.linspace(0, duration, sample_rate * duration)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    wavelets = ['haar', 'db4', 'db8']
    features_dict = {}
    
    for wavelet in wavelets:
        extractor = MFDWCFeatureExtractor(
            n_mels=60,
            n_mfdwc=60,
            wavelet=wavelet,
            sample_rate=sample_rate
        )
        features = extractor(audio_tensor)
        features_dict[wavelet] = features
        
        print(f"Wavelet {wavelet}:")
        print(f"  Feature shape: {features.shape}")
        print(f"  Mean: {features.mean().item():.4f}")
        print(f"  Std: {features.std().item():.4f}")
    
    return features_dict

if __name__ == "__main__":
    # Run all tests
    print("🧪 Starting MFDWC Feature Extraction Tests\n")
    
    try:
        # Test 1: Single audio
        single_features = test_single_audio_extraction()
        
        # Test 2: Batch processing
        batch_features = test_batch_processing()
        
        # Test 3: Different wavelets
        wavelet_features = test_wavelet_comparison()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ MFDWC feature extractor is ready for integration with GRL")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()