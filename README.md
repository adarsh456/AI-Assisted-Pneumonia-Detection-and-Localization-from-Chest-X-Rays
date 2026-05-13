# 🫁 AI-Assisted Pneumonia Detection & Localization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-assisted-pneumonia-detection-and-localization-from-chest-x.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

## 📌 Project Overview
This clinical decision-support tool uses **Deep Learning** to analyze chest X-rays for signs of pneumonia. To build trust with medical professionals, the system goes beyond simple "Normal/Pneumonia" labels by providing **Explainable AI (XAI)** visualizations. 

Using **Grad-CAM (Gradient-weighted Class Activation Mapping)**, the application generates heatmaps that localize the specific regions of the lungs the model identified as suspicious.

## 🚀 Key Features
*   **Automated Diagnosis:** Instant classification of chest X-rays using a high-accuracy CNN model.
*   **Explainable AI (XAI):** Real-time Grad-CAM localization to highlight infected lung tissues.
*   **Interactive UI:** A modern dashboard built with **Streamlit** for seamless image uploading and results visualization.
*   **Clinical Resources:** Integrated educational modules covering symptoms, treatments, and precautions.
*   **Cloud Optimized:** Built for high-performance inference on CPU-based cloud servers.

## 🛠️ Tech Stack
*   **Deep Learning Framework:** TensorFlow 2.18, Keras
*   **Web Framework:** Streamlit
*   **Image Processing:** OpenCV (Headless), PIL (Pillow)
*   **Programming Language:** Python 3.11
*   **Scientific Computing:** NumPy

## 🔬 Model & Architecture
The underlying model is a custom **Convolutional Neural Network (CNN)** structured as follows:
1.  **Feature Extraction:** Multiple blocks of `Conv2D` and `MaxPooling2D` layers to identify spatial patterns in X-ray images.
2.  **Global Pooling:** Efficiently condensing feature maps for the classification head.
3.  **Classification:** Dense layers with Dropout for regularization and a Sigmoid output for binary probability.
4.  **Input Shape:** 150x150x3 (RGB).

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/adarsh456/ai-assisted-pneumonia-detection-and-localization-from-chest-x-rays.git](https://github.com/adarsh456/ai-assisted-pneumonia-detection-and-localization-from-chest-x-rays.git)
   cd ai-assisted-pneumonia-detection-and-localization-from-chest-x-rays

2. Setup the Environment:
It is highly recommended to use Python 3.11 for full compatibility with the project's dependencies, specifically TensorFlow 2.18.
   ```bash
   pip install -r requirements.txt
3. Run the Application:
Launch the local development server:
   ```bash
   streamlit run app.py



##⚠️ Medical Disclaimer

**This application is for educational and research purposes only.** It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 

*   The results provided by this AI are not a medical diagnosis.
*   The Grad-CAM localization is a mathematical representation of model focus and should not be used for clinical or surgical planning.
*   Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition.


