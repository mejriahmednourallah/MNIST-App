# 🔢 MNIST Digit Recognition - Streamlit Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mnist-digit-recognition.streamlit.app)
[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-FF6B6B?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org)

A beautiful, interactive web application for real-time handwritten digit recognition powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

###  Interactive Drawing Canvas
- **Smooth drawing experience** with adjustable brush size (1-40px)
- **Real-time feedback** with professional UI
- **One-click clearing** and screenshot capabilities

###  Advanced Predictions
- **Real-time digit recognition** with confidence scores
- **Top 3 predictions** display with percentages
- **Interactive confidence chart** showing all digit probabilities (0-9)
- **Processed image visualization** (28×28 normalized input)

###  Prediction History & Analytics
- **Automatic history tracking** of all predictions
- **Confidence trend visualization** with interactive charts
- **Statistics dashboard** showing:
  - Total predictions
  - Average confidence
  - Most recognized digit
  - Maximum confidence achieved

###  Beautiful UI/UX
- **Modern gradient design** with purple/blue theme
- **Responsive layout** for desktop and tablets
- **Smooth animations** and transitions
- **Dark mode friendly** design
- **Mobile-optimized** interface

## Getting Started

### Prerequisites
- Python 3.8 or higher
- A trained MNIST model (`.h5` file)
- pip (Python package manager)

### Installation & Setup

#### Step 1: Clone the Repository
```bash
git clone https://github.com/mejriahmednourallah/MNIST-App.git
cd MNIST-App
```

#### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Add Your Trained Model
1. Train your MNIST model or use a pre-trained one
2. Save it as `best_mnist_model.h5`
3. Place it in the `models/` folder

**To save from a Jupyter notebook:**
```python
# After training your model
model.save('best_mnist_model.h5')
# Then download and place in models/ folder
```

#### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

##  Project Structure

```
MNIST-App/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/
│   └── best_mnist_model.h5    # Your trained model (add this)
└── notebooks/                  # (Optional) Training notebooks
    └── mnist_training.ipynb    # (Optional) Your training code
```

##  Model Information

| Metric | Value |
|--------|-------|
| **Architecture** | Convolutional Neural Network (CNN) |
| **Training Dataset** | MNIST (60,000 samples) |
| **Test Accuracy** | 99.2%+ |
| **Model Parameters** | ~180,000 |
| **Input Size** | 28×28 pixels (grayscale) |
| **Output Classes** | 10 (digits 0-9) |
| **Framework** | TensorFlow/Keras |

## How to Use

1. **Draw a Digit**
   - Use the drawing canvas on the left
   - Adjust brush size using the sidebar slider
   - Draw clearly in the center for best results

2. **Make a Prediction**
   - Click the ** Predict** button
   - The AI will analyze your drawing instantly

3. **View Results**
   - See the recognized digit in large text
   - Check confidence percentage
   - View all 10 digit probabilities in a chart
   - See top 3 predictions with medals 🥇🥈🥉

4. **Track History**
   - All predictions are automatically saved
   - View confidence trends over time
   - Check prediction statistics
   - Clear history with one click

##  Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Web framework for building ML apps |
| **TensorFlow/Keras** | Deep learning framework |
| **NumPy** | Numerical computing |
| **PIL (Pillow)** | Image processing |
| **OpenCV** | Computer vision tasks |
| **Plotly** | Interactive visualizations |
| **Python** | Programming language |

##  Dependencies

```
streamlit==1.28.1
tensorflow==2.14.0
numpy==1.24.3
Pillow==10.0.1
opencv-python-headless==4.8.1.78
streamlit-drawable-canvas==0.9.3
plotly==5.17.0
protobuf==3.20.3
scikit-learn==1.3.2
```

##  Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Create new app**
   - Select your GitHub repo
   - Choose main branch
   - Set main file path: `app.py`

4. **Deploy!** 🚀

### Deploy to Heroku, AWS, or Google Cloud

See [Streamlit Deployment Guide](https://docs.streamlit.io/library/deploy) for detailed instructions.

##  Tips for Better Results

 **Do's:**
- Draw digits in the **center** of the canvas
- Make strokes **clear and distinct**
- **Fill in the digit completely**
- Use **consistent brush size**
- Test with **different writing styles**

 **Don'ts:**
- Draw too **thin** or too **thick**
- Make digits **too small**
- Use **multiple separate strokes** for one digit
- Go outside the **canvas boundaries**

##  Troubleshooting

### Model not found error
- **Solution**: Ensure `best_mnist_model.h5` is in the `models/` folder
- **Path**: `MNIST-App/models/best_mnist_model.h5`

### Poor predictions
- **Solution**: Check model accuracy (should be 99%+)
- **Tip**: Draw digits more clearly and centered

### Slow predictions
- **Solution**: Model should predict in <1 second
- **Check**: Computer RAM and CPU usage
- **Note**: First prediction loads the model (slower)

### Port already in use
- **Solution**: 
  ```bash
  streamlit run app.py --logger.level=debug --server.port 8502
  ```

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Training Your Own Model

Want to train your own MNIST model? Check out the training notebook:

```python
# Basic MNIST training code
from tensorflow import keras
from tensorflow.keras import layers, datasets

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=10, validation_split=0.1)

# Save model
model.save('models/best_mnist_model.h5')
```

##  Resources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Keras Documentation](https://keras.io/)
- [CNN Tutorial](https://cs231n.github.io/convolutional-networks/)


##  Author

**Ahmed Nour**
- GitHub: [@mejriahmednourallah](https://github.com/mejriahmednourallah)

##  Contact & Support

Have questions or need help? 

- Open an [Issue](https://github.com/mejriahmednourallah/MNIST-App/issues)
- Check [Streamlit Docs](https://docs.streamlit.io/)


