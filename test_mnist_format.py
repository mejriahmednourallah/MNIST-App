from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST
(X_train, y_train), _ = mnist.load_data()

# Check first image
first_img = X_train[0]

print("="*50)
print("MNIST DATA FORMAT ANALYSIS")
print("="*50)
print(f"Shape: {first_img.shape}")
print(f"Min pixel value: {first_img.min()}")
print(f"Max pixel value: {first_img.max()}")
print(f"Data type: {first_img.dtype}")
print(f"Label: {y_train[0]}")
print()

# Check background vs digit pixels
corners = [first_img[0,0], first_img[0,-1], first_img[-1,0], first_img[-1,-1]]
center = first_img[14,14]

print("Background (corners):", corners, "avg:", np.mean(corners))
print("Center pixel (digit):", center)
print()

if np.mean(corners) < center:
    print("✓ Background is DARK (0), Digits are BRIGHT (255)")
else:
    print("✓ Background is BRIGHT (255), Digits are DARK (0)")

print()
print("Sample of top-left corner (10x10):")
print(first_img[:10, :10])

# Save a sample image
plt.figure(figsize=(5, 5))
plt.imshow(first_img, cmap='gray')
plt.title(f'MNIST Sample: Digit {y_train[0]}')
plt.axis('off')
plt.savefig('mnist_sample.png', bbox_inches='tight', dpi=150)
print("\n✓ Saved sample image as 'mnist_sample.png'")
