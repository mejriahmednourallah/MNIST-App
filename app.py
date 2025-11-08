"""
MNIST Digit Recognition - Streamlit Web App
Author: Ahmed Nour
Description: Draw a digit and get real-time predictions using a trained CNN model
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Elegant beige/cream color palette with smooth animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --cream: #FAF8F3;
        --beige: #E8DCC4;
        --sand: #D4C5A9;
        --warm-brown: #A67B5B;
        --deep-brown: #6B5B4F;
        --accent: #C9A77C;
        --text-dark: #3C3530;
        --text-light: #6B625A;
        --shadow: rgba(107, 91, 79, 0.08);
        --shadow-hover: rgba(107, 91, 79, 0.15);
    }
    
    /* Smooth fade-in animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes gentlePulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, var(--cream) 0%, var(--beige) 100%);
        padding: 2rem 3rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--cream) 0%, var(--beige) 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeIn 0.8s ease-out;
    }
    
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5em;
        font-weight: 700;
        color: var(--deep-brown);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.1em;
        color: var(--text-light);
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Elegant buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--warm-brown) 0%, var(--deep-brown) 100%);
        color: var(--cream);
        border: none;
        padding: 0.85rem 2rem;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1em;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-hover);
        background: linear-gradient(135deg, var(--deep-brown) 0%, var(--warm-brown) 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px var(--shadow);
    }
    
    /* Prediction box - elegant and minimal */
    .prediction-box {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 8px 30px var(--shadow);
        border: 1px solid rgba(166, 123, 91, 0.1);
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
        margin: 1.5rem 0;
    }
    
    .prediction-box:hover {
        box-shadow: 0 12px 40px var(--shadow-hover);
        transform: translateY(-3px);
    }
    
    .prediction-digit {
        font-family: 'Playfair Display', serif;
        font-size: 5em;
        font-weight: 700;
        color: var(--warm-brown);
        margin: 1rem 0;
        animation: gentlePulse 2s ease-in-out infinite;
    }
    
    .confidence-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.3em;
        color: var(--text-dark);
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        color: var(--deep-brown);
        font-size: 1.8em;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        position: relative;
        display: inline-block;
        animation: slideIn 0.6s ease-out;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 50px;
        height: 3px;
        background: var(--accent);
        border-radius: 2px;
    }
    
    /* Canvas container */
    .canvas-container {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px var(--shadow);
        border: 1px solid rgba(166, 123, 91, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.7s ease-out;
        display: inline-block;
        max-width: fit-content;
    }
    
    .canvas-container:hover {
        box-shadow: 0 12px 40px var(--shadow-hover);
    }
    
    /* Fix canvas iframe container to match canvas size */
    .stCustomComponentV1 {
        max-width: 320px !important;
        width: 320px !important;
    }
    
    [data-testid="stElementContainer"] iframe.stCustomComponentV1 {
        max-width: 320px !important;
        width: 320px !important;
    }
    
    /* Make sure canvas parent container doesn't overflow */
    .st-emotion-cache-8atqhb,
    .st-emotion-cache-1vo6xi6 {
        max-width: fit-content !important;
        width: auto !important;
    }
    
    /* Info box */
    .info-box {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        border-left: 4px solid var(--accent);
        box-shadow: 0 4px 20px var(--shadow);
        margin: 1.5rem 0;
        animation: fadeIn 0.6s ease-out;
    }
    
    .info-box h3 {
        font-family: 'Playfair Display', serif;
        color: var(--deep-brown);
        font-size: 1.5em;
        margin-bottom: 1rem;
    }
    
    .info-box p {
        font-family: 'Inter', sans-serif;
        color: var(--text-light);
        line-height: 1.8;
        font-size: 1em;
    }
    
    /* Stat card */
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(166, 123, 91, 0.1);
        margin: 0.5rem 0;
        animation: fadeIn 0.8s ease-out;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px var(--shadow-hover);
    }
    
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.5em;
        font-weight: 700;
        color: var(--warm-brown);
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9em;
        color: var(--text-light);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Top predictions */
    .top-prediction {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 15px var(--shadow);
        border-left: 3px solid var(--accent);
        font-family: 'Inter', sans-serif;
        color: var(--text-dark);
        font-weight: 500;
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    .top-prediction:hover {
        transform: translateX(8px);
        box-shadow: 0 4px 20px var(--shadow-hover);
        border-left-color: var(--warm-brown);
    }
    
    /* Sample cards */
    .sample-card {
        background: white;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px var(--shadow);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border: 1px solid rgba(166, 123, 91, 0.1);
        animation: fadeIn 1s ease-out;
    }
    
    .sample-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 12px 40px var(--shadow-hover);
    }
    
    .sample-card .emoji {
        font-size: 3em;
        margin: 1rem 0;
        filter: grayscale(30%);
        transition: all 0.3s ease;
    }
    
    .sample-card:hover .emoji {
        filter: grayscale(0%);
        transform: scale(1.1);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            var(--accent), 
            transparent);
        margin: 3rem 0;
        opacity: 0.3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--cream) 0%, var(--beige) 100%);
        border-right: 1px solid rgba(166, 123, 91, 0.15);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-dark);
    }
    
    .sidebar-header {
        font-family: 'Playfair Display', serif;
        color: var(--deep-brown);
        font-size: 1.4em;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-stat {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 15px var(--shadow);
        color: var(--text-dark);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid rgba(166, 123, 91, 0.1);
    }
    
    .sidebar-stat:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 20px var(--shadow-hover);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-light);
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(166, 123, 91, 0.15);
        font-family: 'Inter', sans-serif;
    }
    
    .footer a {
        color: var(--warm-brown);
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .footer a:hover {
        color: var(--deep-brown);
    }
    
    /* Slider customization */
    .stSlider > div > div > div > div {
        background: var(--accent) !important;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Element animation */
    .element-container {
        animation: fadeIn 0.8s ease-out;
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
        img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        # DO NOT NORMALIZE - model was trained on raw pixel values [0-255]
        img_array = np.array(img).astype('float32')
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def create_prediction_chart(probabilities):
    """Create a minimalist bar chart"""
    digits = list(range(10))
    probs = probabilities[0] * 100
    
    colors = ['#A67B5B' if i == np.argmax(probs) else '#D4C5A9' for i in range(10)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=probs,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside',
            hovertemplate='<b>Digit %{x}</b><br>%{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Probability Distribution",
            font=dict(size=18, color='#3C3530', family='Playfair Display')
        ),
        xaxis_title="Digit",
        yaxis_title="Confidence (%)",
        height=380,
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(166, 123, 91, 0.1)'
        ),
        plot_bgcolor='rgba(250, 248, 243, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#6B625A', size=11, family='Inter'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def main():
    # Elegant header
    st.markdown("""
        <div class="main-header">
            <h1>Digit Recognition</h1>
            <p>Draw any digit from 0 to 9 and watch the AI recognize it instantly</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="info-box">
            <h3>Model Not Found</h3>
            <p>Please ensure your trained model file <code>best_mnist_model.h5</code> is in the <code>models/</code> folder.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Model Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="sidebar-stat">
                <div style="font-size: 0.85em; opacity: 0.7; margin-bottom: 5px;">Accuracy</div>
                <div style="font-size: 1.6em; font-weight: 700; color: #A67B5B;">99.2%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="sidebar-stat">
                <div style="font-size: 0.85em; opacity: 0.7; margin-bottom: 5px;">Type</div>
                <div style="font-size: 1.6em; font-weight: 700; color: #A67B5B;">CNN</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">Canvas Settings</div>', unsafe_allow_html=True)
        
        stroke_width = st.slider("Brush Size", 5, 35, 18, help="Adjust the thickness of your drawing")
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">Instructions</div>', unsafe_allow_html=True)
        
        st.markdown("""
        1. Draw a digit in the canvas
        2. Click the Predict button
        3. View the AI's prediction
        4. Clear and try again
        
        **Tips for best results:**
        - Draw in the center
        - Use clear, bold strokes
        - Fill the digit completely
        """)
    
    # Main content
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">Draw Here</div>', unsafe_allow_html=True)
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color="#3C3530",
            background_color="#FFFFFF",
            height=320,
            width=320,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Buttons
        col_btn1, col_btn2 = st.columns(2, gap="small")
        
        with col_btn1:
            predict_button = st.button(" Predict", width="stretch", key="predict")
        with col_btn2:
            clear_button = st.button(" Clear Canvas", width="stretch", key="clear")
            if clear_button:
                st.rerun()
    
    with col2:
        st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
        
        if canvas_result.image_data is not None and predict_button:
            if np.sum(canvas_result.image_data[:, :, 3]) == 0:
                st.markdown("""
                <div class="info-box">
                    <h3>Canvas is Empty</h3>
                    <p>Please draw a digit on the canvas first, then click Predict.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing..."):
                    processed_img, display_img = preprocess_image(canvas_result.image_data)
                    
                    if processed_img is not None:
                        predictions = model.predict(processed_img, verbose=0)
                        predicted_digit = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100
                        
                        # Store history
                        st.session_state.prediction_history.append({
                            'digit': predicted_digit,
                            'confidence': confidence,
                            'time': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Display result
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
                            st.caption("Processed Image (28×28)")
                            st.image(display_img, width=110)
                        
                        # Chart
                        st.plotly_chart(create_prediction_chart(predictions), width="stretch")
                        
                        # Top 3
                        st.markdown("**Top 3 Predictions**")
                        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                        
                        for rank, idx in enumerate(top_3_idx, 1):
                            prob = predictions[0][idx] * 100
                            medal = ["🥇", "🥈", "🥉"][rank - 1]
                            st.markdown(f"""
                            <div class="top-prediction">
                                {medal} <strong>Digit {idx}</strong>: {prob:.2f}%
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h3>Ready to Start</h3>
                <p>Draw a digit on the canvas and click the <strong>Predict</strong> button to see the AI in action.</p>
                <p style="margin-top: 1rem; font-size: 0.95em;">Use a thicker brush and draw clearly for the best results.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # History section
    if st.session_state.prediction_history:
        st.markdown('<div class="section-header">Statistics</div>', unsafe_allow_html=True)
        
        all_predictions = [h['digit'] for h in st.session_state.prediction_history]
        all_confidences = [h['confidence'] for h in st.session_state.prediction_history]
        
        stat_cols = st.columns(4, gap="medium")
        
        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(all_predictions)}</div>
                <div class="stat-label">Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[1]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{np.mean(all_confidences):.1f}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[2]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{max(set(all_predictions), key=all_predictions.count)}</div>
                <div class="stat-label">Most Common</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[3]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{max(all_confidences):.1f}%</div>
                <div class="stat-label">Highest</div>
            </div>
            """, unsafe_allow_html=True)
        
        clear_history_btn = st.button("🗑️ Clear History", width="stretch", key="clear_history")
        if clear_history_btn:
            st.session_state.prediction_history = []
            st.rerun()
    
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Sample section
    st.markdown('<div class="section-header">Try These Digits</div>', unsafe_allow_html=True)
    
    sample_cols = st.columns(5, gap="medium")
    samples = [
        {"emoji": "0️⃣", "name": "Zero"},
        {"emoji": "1️⃣", "name": "One"},
        {"emoji": "5️⃣", "name": "Five"},
        {"emoji": "7️⃣", "name": "Seven"},
        {"emoji": "9️⃣", "name": "Nine"}
    ]
    
    for col, sample in zip(sample_cols, samples):
        with col:
            st.markdown(f"""
            <div class="sample-card">
                <div class="emoji">{sample['emoji']}</div>
                <div style="font-weight: 600; color: #6B5B4F; font-size: 1.1em;">{sample['name']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin-bottom: 0.8rem;">Built with Streamlit & TensorFlow</p>
        <p style="font-size: 0.9em; opacity: 0.7;">
            Created by <strong>Ahmed Nour</strong> | 
            <a href="https://github.com/mejriahmednourallah" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()