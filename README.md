🫁 AI-Assisted Pneumonia Detection & Localization
📌 Project Overview
This project is an end-to-end Deep Learning application designed to detect Pneumonia from Chest X-ray images. Beyond simple classification, the app utilizes Grad-CAM (Gradient-weighted Class Activation Mapping) to provide explainability by highlighting the specific regions in the lungs the model used to make its prediction.

This tool is designed as a "second opinion" for healthcare professionals, bridging the gap between high-accuracy AI and clinical trust through visual evidence.

🚀 Key Features
High-Accuracy Classification: Powered by a Custom Convolutional Neural Network (CNN).

Explainable AI (XAI): Integrated Grad-CAM visualization to localize infection areas.

Interactive UI: Built with Streamlit for a seamless, responsive user experience.

Educational Resources: Built-in information regarding symptoms, precautions, and medical insights.

Professional Deployment: Fully optimized for cloud environments with managed dependencies.

🛠️ Tech Stack
Frontend: Streamlit

Deep Learning: TensorFlow 2.18, Keras

Computer Vision: OpenCV, PIL

Data Handling: NumPy, Pandas

Deployment: Streamlit Community Cloud

🔬 Model Architecture
The model is a Sequential CNN consisting of:

Three Convolutional Blocks: Utilizing Conv2D and MaxPooling2D for hierarchical feature extraction.

Dense Layers: For high-level reasoning and binary classification.

Activation: Sigmoid output for precise confidence scoring.

📦 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/adarsh456/ai-assisted-pneumonia-detection-and-localization-from-chest-x-rays.git
cd ai-assisted-pneumonia-detection-and-localization-from-chest-x-rays
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
streamlit run app.py
🖼️ Visualizations & Screenshots
Include a screenshot here of your app showing a successful prediction and the Grad-CAM heatmap.

⚠️ Medical Disclaimer
This application is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

👤 Author
Adarsh

GitHub

LinkedIn
