import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- In a real project, you would load your data here ---
# Example: 
# source_embeddings = np.load('source_features.npy')
# target_embeddings = np.load('target_features.npy')

# For demonstration, let's generate synthetic data
print("Generating synthetic source and target embeddings...")

# Source domain: Two distinct clusters
cluster1_source = np.random.randn(50, 128) + np.array([5, 5] + [0]*126)
cluster2_source = np.random.randn(50, 128) + np.array([-5, -5] + [0]*126)
source_embeddings = np.vstack([cluster1_source, cluster2_source])

# Target domain: Similar clusters but with a noticeable domain shift
domain_shift = np.array([2, -2] + [0]*126) # The shift
cluster1_target = np.random.randn(50, 128) + np.array([5, 5] + [0]*126) + domain_shift
cluster2_target = np.random.randn(50, 128) + np.array([-5, -5] + [0]*126) + domain_shift
target_embeddings = np.vstack([cluster1_target, cluster2_target])

print(f"Shape of source embeddings: {source_embeddings.shape}")
print(f"Shape of target embeddings: {target_embeddings.shape}")


# Combine embeddings into a single array
combined_embeddings = np.vstack((source_embeddings, target_embeddings))

# Create labels: 0 for source, 1 for target
num_source = source_embeddings.shape[0]
num_target = target_embeddings.shape[0]
labels = np.array([0] * num_source + [1] * num_target)


print("Running t-SNE... (This may take a moment)")

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
embeddings_2d = tsne.fit_transform(combined_embeddings)

print("t-SNE finished.")


# Set up the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

# Separate the 2D points by their domain
source_points = embeddings_2d[labels == 0]
target_points = embeddings_2d[labels == 1]

# Plot source points in blue
ax.scatter(source_points[:, 0], source_points[:, 1], c='blue', label='Source Domain', alpha=0.7)

# Plot target points in red
ax.scatter(target_points[:, 0], target_points[:, 1], c='red', label='Target Domain', alpha=0.7)

# Add title and legend
ax.set_title('t-SNE Visualization of Source vs. Target Domains', fontsize=16)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend()
ax.grid(True)

plt.show()