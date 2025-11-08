from keras.datasets import mnist
import numpy as np

# Load MNIST
(X_train, y_train), _ = mnist.load_data()

# Check first few images
for i in range(3):
    img = X_train[i]
    print("="*50)
    print(f"IMAGE {i}: MNIST DATA FORMAT ANALYSIS (Label: {y_train[i]})")
    print("="*50)
    print(f"Shape: {img.shape}")
    print(f"Min pixel value: {img.min()}")
    print(f"Max pixel value: {img.max()}")
    print(f"Data type: {img.dtype}")
    print()
    
    # Check background vs digit pixels
    corners = [img[0,0], img[0,-1], img[-1,0], img[-1,-1]]
    center = img[14,14]
    
    print("Background (corners):", corners, "avg:", np.mean(corners))
    print("Center pixel (digit area):", center)
    print()
    
    if np.mean(corners) < center:
        print("✓ Background is DARK (low values), Digits are BRIGHT (high values)")
    else:
        print("✓ Background is BRIGHT (high values), Digits are DARK (low values)")
    
    print()
    print("Sample of top-left corner (8x8):")
    print(img[:8, :8])
    print()
    print("Sample of center area (8x8):")
    print(img[10:18, 10:18])
    print("\n")
