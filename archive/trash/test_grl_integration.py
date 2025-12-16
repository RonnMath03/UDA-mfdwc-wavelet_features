"""
Test script to verify GRL-MFDWC integration before full training
"""

import torch
import numpy as np
from mfdwc_extractor import MFDWCFeatureExtractor
from mfdwc_grl import (
    GradientReversalLayer, Classifier, Discriminator, 
    MFDWC_CONFIG, DEVICE
)

def test_mfdwc_integration():
    """Test MFDWC feature extraction integration"""
    print("=== Testing MFDWC Integration ===")
    
    # Create test audio batch
    batch_size = 4
    audio_length = MFDWC_CONFIG['sample_rate'] * 10  # 10 seconds
    test_audio = torch.randn(batch_size, audio_length).to(DEVICE)
    
    # Initialize MFDWC extractor with corrected configuration
    feature_extractor = MFDWCFeatureExtractor(**MFDWC_CONFIG).to(DEVICE)
    
    try:
        # Test feature extraction
        features = feature_extractor(test_audio)
        print(f"✅ Feature extraction successful: {test_audio.shape} -> {features.shape}")
        
        feature_dim = features.shape[1]
        print(f"Feature dimension: {feature_dim}")
        
        return features, feature_dim
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None, None

def test_grl_models(features, feature_dim, num_classes=10):
    """Test GRL models with extracted features"""
    print("\n=== Testing GRL Models ===")
    
    try:
        # Initialize models
        classifier = Classifier(input_size=feature_dim, num_classes=num_classes).to(DEVICE)
        discriminator = Discriminator(input_size=feature_dim).to(DEVICE)
        grl = GradientReversalLayer().to(DEVICE)
        
        print(f"✅ Models initialized successfully")
        
        # Test forward pass
        with torch.no_grad():
            # Classification
            class_outputs = classifier(features)
            print(f"✅ Classification: {features.shape} -> {class_outputs.shape}")
            
            # Domain discrimination with GRL
            grl_features = grl(features)
            domain_outputs = discriminator(grl_features)
            print(f"✅ Domain discrimination: {features.shape} -> {domain_outputs.shape}")
            
            # Check output ranges
            class_probs = torch.softmax(class_outputs, dim=1)
            print(f"Class probabilities range: [{class_probs.min():.4f}, {class_probs.max():.4f}]")
            print(f"Domain outputs range: [{domain_outputs.min():.4f}, {domain_outputs.max():.4f}]")
        
        return classifier, discriminator, grl
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return None, None, None

def test_training_step(feature_extractor, classifier, discriminator, grl):
    """Test a single training step"""
    print("\n=== Testing Training Step ===")
    
    try:
        # Create mock data
        batch_size = 4
        audio_length = MFDWC_CONFIG['sample_rate'] * 10
        num_classes = 10
        
        # Source domain data (device 'a' = 1)
        source_audio = torch.randn(batch_size, audio_length).to(DEVICE)
        source_labels = torch.randint(0, num_classes, (batch_size,)).to(DEVICE)
        source_domains = torch.ones(batch_size).to(DEVICE)
        
        # Target domain data (device 'b', 'c' = 0)
        target_audio = torch.randn(batch_size, audio_length).to(DEVICE)
        target_domains = torch.zeros(batch_size).to(DEVICE)
        
        # Set models to training mode
        feature_extractor.train()
        classifier.train()
        discriminator.train()
        
        # Set GRL lambda
        grl.set_lambda(0.5)
        
        # Initialize optimizers (MFDWC extractor has no trainable parameters)
        import torch.optim as optim
        
        # Check if feature extractor has parameters
        extractor_params = list(feature_extractor.parameters())
        if len(extractor_params) > 0:
            optimizer_F = optim.Adam(feature_extractor.parameters(), lr=0.0001)
        else:
            optimizer_F = None  # No trainable parameters in MFDWC
            print("Note: MFDWC feature extractor has no trainable parameters (uses signal processing)")
        
        optimizer_C = optim.Adam(classifier.parameters(), lr=0.0001)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
        
        # Loss functions
        class_criterion = torch.nn.CrossEntropyLoss()
        domain_criterion = torch.nn.BCELoss()
        
        # Forward pass - Source
        source_features = feature_extractor(source_audio)
        source_class_outputs = classifier(source_features)
        source_domain_features = grl(source_features)
        source_domain_outputs = discriminator(source_domain_features).squeeze()
        
        # Forward pass - Target
        target_features = feature_extractor(target_audio)
        target_domain_features = grl(target_features)
        target_domain_outputs = discriminator(target_domain_features).squeeze()
        
        # Calculate losses
        class_loss = class_criterion(source_class_outputs, source_labels)
        domain_loss_source = domain_criterion(source_domain_outputs, source_domains)
        domain_loss_target = domain_criterion(target_domain_outputs, target_domains)
        domain_loss = domain_loss_source + domain_loss_target
        total_loss = class_loss + domain_loss
        
        print(f"✅ Loss calculation successful:")
        print(f"  Classification loss: {class_loss.item():.4f}")
        print(f"  Domain loss: {domain_loss.item():.4f}")
        print(f"  Total loss: {total_loss.item():.4f}")
        
        # Backward pass
        if optimizer_F is not None:
            optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        optimizer_D.zero_grad()
        
        total_loss.backward()
        
        if optimizer_F is not None:
            optimizer_F.step()
        optimizer_C.step()
        optimizer_D.step()
        
        print(f"✅ Backward pass and optimization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_parameters():
    """Test model parameter counts"""
    print("\n=== Testing Model Parameters ===")
    
    feature_extractor = MFDWCFeatureExtractor(**MFDWC_CONFIG)
    
    # Get feature dimension
    test_audio = torch.randn(1, MFDWC_CONFIG['sample_rate'] * 10)
    with torch.no_grad():
        test_features = feature_extractor(test_audio)
        feature_dim = test_features.shape[1]
    
    classifier = Classifier(input_size=feature_dim, num_classes=10)
    discriminator = Discriminator(input_size=feature_dim)
    
    # Count parameters
    feature_params = sum(p.numel() for p in feature_extractor.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    discriminator_params = sum(p.numel() for p in discriminator.parameters())
    total_params = feature_params + classifier_params + discriminator_params
    
    print(f"Feature Extractor parameters: {feature_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Discriminator parameters: {discriminator_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Feature dimension: {feature_dim}")
    
    return total_params

def main():
    """Run all integration tests"""
    print("🧪 Starting GRL-MFDWC Integration Tests\n")
    print(f"Device: {DEVICE}")
    print(f"MFDWC Config: {MFDWC_CONFIG}")
    print("-" * 60)
    
    # Test 1: MFDWC feature extraction
    features, feature_dim = test_mfdwc_integration()
    if features is None:
        print("❌ Integration tests failed at feature extraction")
        return
    
    # Test 2: GRL models
    classifier, discriminator, grl = test_grl_models(features, feature_dim)
    if classifier is None:
        print("❌ Integration tests failed at model testing")
        return
    
    # Test 3: Training step
    feature_extractor = MFDWCFeatureExtractor(**MFDWC_CONFIG).to(DEVICE)
    success = test_training_step(feature_extractor, classifier, discriminator, grl)
    if not success:
        print("❌ Integration tests failed at training step")
        return
    
    # Test 4: Model parameters
    total_params = test_model_parameters()
    
    print("\n" + "="*60)
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print(f"✅ MFDWC feature extraction: Working")
    print(f"✅ GRL models: Working")
    print(f"✅ Training step: Working")
    print(f"✅ Total model parameters: {total_params:,}")
    print(f"✅ Feature dimension: {feature_dim}")
    print("\n🚀 Ready for full training!")

if __name__ == "__main__":
    main()