## CNN Integration Guidelines

### Feature Shape for CNN Input

MFDWC produces **1D feature vectors**, which need to be reshaped for CNN processing:

```python
# MFDWC output: (batch_size, 90)
features = extractor(audio_batch)  # Shape: (32, 90)

# Reshape for CNN (add channel dimension)
features_cnn = features.unsqueeze(1)  # Shape: (32, 1, 90)

# Or reshape as 2D for Conv2D
features_2d = features.view(batch_size, 1, 9, 10)  # Example: (32, 1, 9, 10)
```

### CNN Architecture Example 1: 1D CNN

```python
import torch.nn as nn

class MFDWC_CNN_1D(nn.Module):
    def __init__(self, num_classes=10):
        super(MFDWC_CNN_1D, self).__init__()
        
        # MFDWC feature extractor
        self.mfdwc = MFDWCFeatureExtractor(
            n_mels=60,
            n_mfdwc=30,
            wavelet='haar',
            sample_rate=32000
        )
        
        # 1D CNN layers
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, audio):
        # Extract MFDWC features
        features = self.mfdwc(audio)  # (batch, 90)
        
        # Reshape for Conv1D: (batch, channels, length)
        x = features.unsqueeze(1)  # (batch, 1, 90)
        
        # Apply CNN
        x = self.conv1d(x)  # (batch, 128, 1)
        x = x.squeeze(-1)   # (batch, 128)
        
        # Classify
        output = self.classifier(x)
        
        return output

# Usage
model = MFDWC_CNN_1D(num_classes=10)
audio_batch = torch.randn(32, 320000)  # 32 samples, 10 seconds each
output = model(audio_batch)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### CNN Architecture Example 2: 2D CNN

```python
class MFDWC_CNN_2D(nn.Module):
    def __init__(self, num_classes=10):
        super(MFDWC_CNN_2D, self).__init__()
        
        # MFDWC feature extractor
        self.mfdwc = MFDWCFeatureExtractor(
            n_mels=60,
            n_mfdwc=30,
            wavelet='haar',
            sample_rate=32000
        )
        
        # 2D CNN layers
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, audio):
        # Extract MFDWC features
        features = self.mfdwc(audio)  # (batch, 90)
        
        # Reshape for Conv2D: (batch, 1, height, width)
        # Example: reshape 90 features to 9x10 grid
        batch_size = features.shape[0]
        x = features.view(batch_size, 1, 9, 10)  # (batch, 1, 9, 10)
        
        # Apply CNN
        x = self.conv2d(x)  # (batch, 128, 1, 1)
        x = x.view(batch_size, -1)  # (batch, 128)
        
        # Classify
        output = self.classifier(x)
        
        return output

# Usage
model = MFDWC_CNN_2D(num_classes=10)
audio_batch = torch.randn(32, 320000)
output = model(audio_batch)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### CNN Architecture Example 3: Hybrid (MFDWC + Learnable Layers)

```python
class MFDWC_Hybrid_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MFDWC_Hybrid_CNN, self).__init__()
        
        # MFDWC feature extractor (frozen)
        self.mfdwc = MFDWCFeatureExtractor(
            n_mels=60,
            n_mfdwc=30,
            wavelet='haar',
            sample_rate=32000
        )
        
        # Feature enhancement layers
        self.feature_enhance = nn.Sequential(
            nn.Linear(90, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )
        
        # Reshape for CNN
        self.reshape_h = 8
        self.reshape_w = 16
        
        # 2D CNN
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, audio):
        # Extract MFDWC features
        features = self.mfdwc(audio)  # (batch, 90)
        
        # Enhance features
        features_enhanced = self.feature_enhance(features)  # (batch, 128)
        
        # Reshape for Conv2D
        batch_size = features_enhanced.shape[0]
        x = features_enhanced.view(batch_size, 1, self.reshape_h, self.reshape_w)
        
        # Apply CNN
        x = self.conv2d(x)
        x = x.view(batch_size, -1)
        
        # Classify
        output = self.classifier(x)
        
        return output
```

### Training Loop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Initialize model
model = MFDWC_CNN_1D(num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(audio)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Epoch statistics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Train Accuracy: {accuracy:.2f}%")
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_accuracy = 100. * val_correct / val_total
    print(f"  Val Accuracy: {val_accuracy:.2f}%")
```

---