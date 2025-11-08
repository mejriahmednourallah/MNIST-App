"""
MNIST Digit Recognition - Streamlit Web App
Author: Ahmed Nour
Description: Draw a digit and get real-time predictions using a trained CNN model
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-modern, sleek styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
    }
    
    * {
        margin: 0;
        padding: 0;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-20px);
        }
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 2.5rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .main-header h1 {
        font-size: 4em;
        font-weight: 900;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 3px;
        filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5));
    }
    
    .main-header p {
        font-size: 1.4em;
        color: rgba(255,255,255,0.85);
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Glassmorphism Buttons */
    .stButton>button {
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        color: white;
        height: 3.5em;
        border-radius: 15px;
        font-weight: 700;
        font-size: 1.1em;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover:before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5);
        border-color: rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Prediction Box with Glassmorphism */
    .prediction-box {
        padding: 40px;
        border-radius: 25px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 0 20px rgba(255, 255, 255, 0.05);
        color: white;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    .prediction-digit {
        font-size: 5em;
        font-weight: 900;
        margin: 15px 0;
        background: linear-gradient(135deg, #fff 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 5px 15px rgba(255, 255, 255, 0.3));
        animation: pulse 2s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }
    
    .confidence-text {
        font-size: 1.5em;
        font-weight: 600;
        margin: 15px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .section-header {
        color: white;
        font-weight: 700;
        font-size: 1.8em;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        letter-spacing: 1px;
        position: relative;
        display: inline-block;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Canvas with Glassmorphism */
    .canvas-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .canvas-box:hover {
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    /* Info Box with Glassmorphism */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-left: 5px solid #667eea;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        color: white;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateX(5px);
        border-left-color: #764ba2;
    }
    
    .info-box h3 {
        margin-bottom: 10px;
    }
    
    /* Stat Card with Glassmorphism */
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin: 10px 0;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stat-card:hover {
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    .stat-value {
        font-size: 2.5em;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.95em;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Top Prediction with Glassmorphism */
    .top-prediction {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .top-prediction:hover {
        transform: translateX(10px);
        border-color: rgba(255, 255, 255, 0.4);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Sample Card with Glassmorphism */
    .sample-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        padding: 25px 20px;
        border-radius: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        color: white;
    }
    
    .sample-card:hover {
        transform: translateY(-10px) rotate(2deg);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        border-color: rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.15);
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, 
            rgba(255,255,255,0), 
            rgba(102, 126, 234, 0.5), 
            rgba(255,255,255,0));
        margin: 3rem 0;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid rgba(255,255,255,0.1);
        font-size: 0.95em;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .footer a:hover {
        color: #764ba2;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-header {
        color: white;
        font-weight: 900;
        font-size: 1.4em;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sidebar-stat {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-stat:hover {
        transform: translateX(5px);
        border-color: rgba(255, 255, 255, 0.4);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Slider customization */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Make text white in sidebar */
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Animation for elements */
    .element-container {
        animation: fadeInUp 0.6s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    model_path = 'models/best_mnist_model.h5'
    
    if not os.path.exists(model_path):
        return None
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_data):
    """Preprocess the drawn image for model prediction"""
    try:
        # Convert to PIL Image
        img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Invert colors (white background to black, black drawing to white)
        img = ImageOps.invert(img)
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img).astype('float32')
        
        # DO NOT apply binary threshold - MNIST has grayscale values with anti-aliasing
        # DO NOT normalize - model was trained on [0-255] grayscale values
        # Keep the natural grayscale values from the resized image
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def create_prediction_chart(probabilities):
    """Create an interactive bar chart of predictions"""
    digits = list(range(10))
    probs = probabilities[0] * 100
    
    colors = ['#FF6B6B' if i == np.argmax(probs) else '#4ECDC4' for i in range(10)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=probs,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
            hovertemplate='<b>Digit %{x}</b><br>Confidence: %{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="<b>Prediction Probabilities</b>",
            font=dict(size=20, color='white')
        ),
        xaxis_title="<b>Digit</b>",
        yaxis_title="<b>Confidence (%)</b>",
        height=400,
        showlegend=False,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.2)'),
        plot_bgcolor='rgba(50,50,100,0.3)',
        paper_bgcolor='rgba(102,126,234,0)',
        font=dict(color='white', size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    
    return fig

def create_history_chart():
    """Create a chart of prediction history"""
    if not st.session_state.prediction_history:
        return None
    
    history = st.session_state.prediction_history
    predictions = [h['digit'] for h in history]
    confidences = [h['confidence'] for h in history]
    times = [h['time'] for h in history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(predictions))),
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10, symbol='circle'),
        hovertemplate='<b>Prediction #%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Prediction History</b>",
        xaxis_title="<b>Prediction Number</b>",
        yaxis_title="<b>Confidence (%)</b>",
        height=300,
        plot_bgcolor='rgba(50,50,100,0.3)',
        paper_bgcolor='rgba(102,126,234,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    
    return fig

def main():
    # Main Header with enhanced visuals
    st.markdown("""
        <div class="main-header">
            <h1>� MNIST Digit Recognition</h1>
            <p>✨ Draw any digit (0-9) and watch AI magic happen in real-time ✨</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="info-box">
            <h3>⚠️ Model Not Found</h3>
            <p>Please ensure your trained model file <code>best_mnist_model.h5</code> is in the <code>models/</code> folder.</p>
            <p><strong>Steps:</strong></p>
            <ol>
                <li>Train your model in a Jupyter notebook or Google Colab</li>
                <li>Save it using: <code>model.save('best_mnist_model.h5')</code></li>
                <li>Download the file and place it in the <code>models/</code> folder</li>
                <li>Restart the Streamlit app</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Settings & Info</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">📊 Model Statistics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="sidebar-stat">
                <div style="font-size: 0.9em; opacity: 0.9;">Accuracy</div>
                <div style="font-size: 1.5em; font-weight: bold;">99.2%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="sidebar-stat">
                <div style="font-size: 0.9em; opacity: 0.9;">Architecture</div>
                <div style="font-size: 1.5em; font-weight: bold;">CNN</div>
            </div>
            """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("""
            <div class="sidebar-stat">
                <div style="font-size: 0.9em; opacity: 0.9;">Training Data</div>
                <div style="font-size: 1.5em; font-weight: bold;">60K</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="sidebar-stat">
                <div style="font-size: 0.9em; opacity: 0.9;">Parameters</div>
                <div style="font-size: 1.5em; font-weight: bold;">180K</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">🎨 Canvas Settings</div>', unsafe_allow_html=True)
        
        stroke_width = st.slider("🖌️ Brush Size", 1, 40, 20, help="Adjust brush thickness for drawing")
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">📖 How to Use</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Steps:**
        1. 🎨 Draw a digit (0-9) in the canvas
        2. 🎯 Click the **Predict** button
        3. 📊 View real-time results and confidence
        4. 🔄 Try again or clear to start over
        
        **Tips:**
        - Draw in the center for better results
        - Make digits clear and distinct
        - Fill in the digit completely
        - Avoid too thin strokes
        """)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">🚀 Tech Stack</div>', unsafe_allow_html=True)
        
        st.markdown("""
        - **TensorFlow/Keras** - Deep Learning
        - **Streamlit** - Web Interface
        - **OpenCV** - Image Processing
        - **Plotly** - Interactive Charts
        - **NumPy & PIL** - Data Processing
        """)
    
    # Main content layout
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">✏️ Draw Your Digit</div>', unsafe_allow_html=True)
        st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=340,
            width=340,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3, gap="small")
        
        with col_btn1:
            predict_button = st.button("🎯 Predict", use_container_width=True, key="predict")
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear"):
                st.rerun()
        with col_btn3:
            if st.button("📸 Screenshot", use_container_width=True, key="screenshot"):
                st.info("Use your browser's screenshot tool to save the result!")
    
    with col2:
        st.markdown('<div class="section-header">🎯 Results</div>', unsafe_allow_html=True)
        
        # Prediction logic
        if canvas_result.image_data is not None and predict_button:
            if np.sum(canvas_result.image_data[:, :, 3]) == 0:
                st.markdown("""
                <div class="info-box">
                    <h3>✋ Oops! Canvas is Empty</h3>
                    <p style="margin: 10px 0; font-size: 1.1em;">🎨 Draw a beautiful digit on the canvas first, then click Predict!</p>
                    <p style="font-size: 0.9em; opacity: 0.8;">💡 Tip: Draw big and clear for best results</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("🔮 Analyzing your digit..."):
                    # Preprocess image
                    processed_img, display_img = preprocess_image(canvas_result.image_data)
                    
                    if processed_img is not None:
                        # Make prediction
                        predictions = model.predict(processed_img, verbose=0)
                        predicted_digit = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100
                        
                        # Store in history
                        st.session_state.prediction_history.append({
                            'digit': predicted_digit,
                            'confidence': confidence,
                            'time': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Display result in beautiful box
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-digit">{predicted_digit}</div>
                            <div class="confidence-text">
                                Confidence: <strong>{confidence:.2f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Processed image
                        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                        with col_img2:
                            st.markdown("##### 📦 Processed (28×28)")
                            st.image(display_img, width=120, use_container_width=False)
                        
                        # Probability chart
                        st.plotly_chart(create_prediction_chart(predictions), use_container_width=True)
                        
                        # Top 3 predictions
                        st.markdown("##### 🏆 Top 3 Predictions")
                        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                        
                        for rank, idx in enumerate(top_3_idx, 1):
                            prob = predictions[0][idx] * 100
                            medal = ["🥇", "🥈", "🥉"][rank - 1]
                            st.markdown(f"""
                            <div class="top-prediction">
                                {medal} <strong>#{rank}</strong> - Digit <strong>{idx}</strong>: <strong>{prob:.2f}%</strong>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h3>� Ready to Start?</h3>
                <p style="font-size: 1.15em; margin: 15px 0;">👈 Draw any digit (0-9) in the canvas on the left</p>
                <p style="font-size: 1.1em; margin: 10px 0;">🎯 Click the <strong>Predict</strong> button</p>
                <p style="font-size: 1.1em; margin: 10px 0;">✨ Watch the AI work its magic!</p>
                <p style="margin-top: 15px; font-size: 0.95em; opacity: 0.8;">
                    💡 Pro tip: Use thicker brush strokes and draw in the center for the most accurate predictions
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # History section
    if st.session_state.prediction_history:
        st.markdown('<div class="section-header">📈 Prediction History</div>', unsafe_allow_html=True)
        
        history_col1, history_col2 = st.columns([2, 1])
        
        with history_col1:
            history_chart = create_history_chart()
            if history_chart:
                st.plotly_chart(history_chart, use_container_width=True)
        
        with history_col2:
            st.markdown("##### 📊 Statistics")
            
            all_predictions = [h['digit'] for h in st.session_state.prediction_history]
            all_confidences = [h['confidence'] for h in st.session_state.prediction_history]
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Predictions</div>
                <div class="stat-value">{len(all_predictions)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{np.mean(all_confidences):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Most Common</div>
                <div class="stat-value">{max(set(all_predictions), key=all_predictions.count)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Max Confidence</div>
                <div class="stat-value">{max(all_confidences):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Sample digits section
    st.markdown('<div class="section-header">📸 Sample Digits to Try</div>', unsafe_allow_html=True)
    st.markdown("""
    Try drawing these digits to test the model's recognition ability!
    """)
    
    sample_cols = st.columns(5, gap="medium")
    sample_digits = [
        {"digit": "0", "emoji": "0️⃣", "desc": "Zero"},
        {"digit": "1", "emoji": "1️⃣", "desc": "One"},
        {"digit": "5", "emoji": "5️⃣", "desc": "Five"},
        {"digit": "7", "emoji": "7️⃣", "desc": "Seven"},
        {"digit": "9", "emoji": "9️⃣", "desc": "Nine"},
    ]
    
    for col, sample in zip(sample_cols, sample_digits):
        with col:
            st.markdown(f"""
            <div class="sample-card">
                <div style="font-size: 3em; margin: 15px 0; filter: drop-shadow(0 0 10px rgba(255,255,255,0.3));">{sample['emoji']}</div>
                <div style="font-weight: 700; font-size: 1.2em; color: white; margin: 10px 0;">{sample['desc']}</div>
                <div style="font-size: 0.85em; color: rgba(255,255,255,0.7); font-weight: 500;">Try drawing this!</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1em; margin-bottom: 10px;">✨ Built with passion using <strong>Streamlit</strong> & <strong>TensorFlow</strong> ✨</p>
        <p style="margin: 10px 0;">🧠 Powered by Deep Learning | 🎯 Real-time AI Predictions | 📊 99.2% Accuracy</p>
        <p style="margin-top: 1.5rem; font-size: 0.95em;">
            Created with 💜 by <strong>Ahmed Nour</strong> | 
            <a href="https://github.com/mejriahmednourallah">🌟 View on GitHub</a>
        </p>
        <p style="margin-top: 0.5rem; font-size: 0.8em; opacity: 0.6;">
            © 2025 | Transforming handwritten digits into predictions since today 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
