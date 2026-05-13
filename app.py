import io
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
from pathlib import Path
import time
import base64
from typing import Tuple, Optional
import cv2  # <-- ADDED for Grad-CAM


@st.cache_resource(show_spinner=False)
def load_model():
    # 1. Load the Keras model file
    model = tf.keras.models.load_model("pneumonia_detection_model.h5")
    
    # 2. Explicitly build the model with the expected input shape
    # This ensures every layer (conv2d, dense, etc.) has defined output tensors
    model.build((None, 150, 150, 3)) 
    
    return model


def preprocess_image(image_file: bytes) -> np.ndarray:
	"""Preprocess uploaded image to match training: 150x150 RGB, scaled 1/255."""
	try:
		# Reset file pointer to beginning
		if hasattr(image_file, 'seek'):
			image_file.seek(0)
		
		# Open and process image
		image = Image.open(io.BytesIO(image_file)).convert("RGB")
		image = image.resize((150, 150), Image.Resampling.LANCZOS)
		arr = np.asarray(image, dtype=np.float32) / 255.0
		arr = np.expand_dims(arr, axis=0)
		return arr
	except Exception as e:
		st.error(f"Error processing image: {str(e)}")
		st.error(f"Image file type: {type(image_file)}")
		return None


def get_prediction_label_and_prob(score: float, threshold: float) -> tuple[str, float]:
    """Map sigmoid score to label and probability for the predicted class using a threshold."""
    if score >= threshold:
        label = "Pneumonia"
        probability = float(score)
    else:
        label = "Normal"
        probability = float(1.0 - score)
    # Cap extremes for display to avoid overconfidence appearance
    probability = float(max(0.05, min(0.95, probability)))
    return label, probability


def get_confidence_level(score: float, threshold: float) -> tuple[str, float]:
    """Return qualitative confidence level and a 0-1 strength based on distance from threshold."""
    margin = abs(score - threshold)
    # Normalize margin to [0, 1] assuming max meaningful margin ~0.5
    strength = min(1.0, margin / 0.5)
    if margin >= 0.25:
        level = "High"
    elif margin >= 0.12:
        level = "Medium"
    else:
        level = "Low"
    return level, strength


def validate_image_file(uploaded_file) -> bool:
	"""Validate uploaded image file."""
	try:
		# Check if file exists and has content
		if not uploaded_file or uploaded_file.size == 0:
			st.error("❌ Empty file uploaded. Please select a valid image.")
			return False
		
		# Check file size (10MB limit)
		if uploaded_file.size > 10 * 1024 * 1024:
			st.error("⚠️ File size too large. Please upload an image smaller than 10MB.")
			return False
		
		# Check file type
		allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
		if uploaded_file.type not in allowed_types:
			st.error(f"❌ Unsupported file type: {uploaded_file.type}. Please upload JPG, JPEG, or PNG.")
			return False
		
		# Try to open the image to validate it's a valid image
		uploaded_file.seek(0)
		image = Image.open(uploaded_file)
		image.verify()  # Verify it's a valid image
		uploaded_file.seek(0)  # Reset for later use
		
		return True
	except Exception as e:
		st.error(f"❌ Invalid image file: {str(e)}")
		return False


def format_probability(p: float) -> str:
	return f"{p * 100:.2f}%"


def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """
    Robust Grad-CAM implementation for all Keras models after loading: always uses model.get_layer and model.inputs.
    """
    import cv2
    from tensorflow.keras.layers import Conv2D
    try:
        # Identify available Conv2D layers
        conv_layers = [(i, layer.name) for i, layer in enumerate(model.layers) if isinstance(layer, Conv2D)]
        if not conv_layers:
            return None, "Model contains no Conv2D layers."
        # Auto-select last Conv2D if not explicitly provided
        if last_conv_layer_name is None:
            _, layer_to_try = conv_layers[-1]
        else:
            candidates = [item for item in conv_layers if item[1] == last_conv_layer_name]
            if not candidates:
                return None, f"Named Conv2D layer '{last_conv_layer_name}' not found. Available: {[name for _, name in conv_layers]}"
            _, layer_to_try = candidates[0]
        # Use get_layer (.output) and model.inputs for maximum compatibility
        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(layer_to_try).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0
            class_channel = predictions[:, pred_index] if predictions.shape[-1] > 1 else predictions[:, 0]
            grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return None, "Could not compute gradients wrt convolutional layer. Check model connectivity."
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1.0
        heatmap = heatmap / max_val
        heatmap = cv2.resize(heatmap.numpy(), (150, 150))
        return heatmap.astype(np.float32), None
    except Exception as e:
        conv_layers = [(layer.name, type(layer).__name__) for layer in model.layers]
        return None, f"Error in Grad-CAM: {str(e)}. All layers: {conv_layers}"


def overlay_heatmap(img_bytes, heatmap, intensity=0.5):
    """
    Overlays the Grad-CAM heatmap onto the input image, returns the result as a PIL Image.
    Args:
        img_bytes: Original image bytes (JPG/PNG).
        heatmap: np.ndarray, shape (150, 150) normalized [0, 1].
        intensity: blending factor, float in [0, 1].
    Returns:
        pil_img: PIL.Image.Image instance of the overlay, RGB.
        error_message: None if ok, else string describing the error.
    """
    try:
        # Open and resize to 150x150 for overlay
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((150, 150), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        # Colorize heatmap (OpenCV uses BGR by default)
        colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(colored_heatmap, intensity, img_array, 1 - intensity, 0)
        pil_img = Image.fromarray(superimposed_img)
        return pil_img, None
    except Exception as e:
        return None, f"Could not overlay Grad-CAM: {str(e)}"


def main():
	st.set_page_config(
		page_title="🫁 Pneumonia Detection App",
		page_icon="🫁",
		layout="wide",
		initial_sidebar_state="collapsed",
		menu_items={
			'Get Help': 'https://streamlit.io',
			'Report a bug': None,
			'About': "AI-powered pneumonia detection using chest X-rays"
		}
	)
	
	# Hide the default Streamlit header
	st.markdown("""
		<style>
			#MainMenu {visibility: hidden;}
			footer {visibility: hidden;}
			header {visibility: hidden;}
		</style>
	""", unsafe_allow_html=True)

	# UI state for expand/collapse behavior of info section
	if "expand_all" not in st.session_state:
		st.session_state["expand_all"] = False
	# Analysis workflow state
	if "analyzed" not in st.session_state:
		st.session_state["analyzed"] = False
	if "uploaded_bytes" not in st.session_state:
		st.session_state["uploaded_bytes"] = None
	if "last_score" not in st.session_state:
		st.session_state["last_score"] = None
	if "current_file" not in st.session_state:
		st.session_state["current_file"] = None

	# Enhanced Header with Animations
	st.markdown(
		"""
		<style>
			@keyframes gradientShift {
				0% { background-position: 0% 50%; }
				50% { background-position: 100% 50%; }
				100% { background-position: 0% 50%; }
			}
			
			@keyframes pulse {
				0% { transform: scale(1); }
				50% { transform: scale(1.05); }
				100% { transform: scale(1); }
			}
			
			@keyframes fadeInUp {
				from { opacity: 0; transform: translateY(30px); }
				to { opacity: 1; transform: translateY(0); }
			}
			
			html, body, .main { 
				background: linear-gradient(-45deg, #0f172a, #111827, #0b1020, #1e293b);
				background-size: 400% 400%;
				animation: gradientShift 15s ease infinite;
			}
			.main > div { 
				padding-top: 4rem; 
				padding-left: 1rem;
			}
			
			/* Fix for sidebar overlap */
			.stApp > header {
				background-color: transparent;
			}
			
			.stApp {
				margin-top: 0;
			}
			
			/* Ensure content doesn't hide behind sidebar */
			.block-container {
				padding-top: 2rem !important;
				padding-left: 1rem !important;
			}
			
			.app-title { 
				margin-top: 2rem; 
				margin-bottom: 1rem; 
				font-size: 2.5rem; 
				font-weight: 800; 
				letter-spacing: .5px; 
				background: linear-gradient(45deg, #6c63ff, #00c2ff, #ff6b6b);
				background-size: 200% 200%;
				animation: gradientShift 3s ease infinite;
				-webkit-background-clip: text;
				-webkit-text-fill-color: transparent;
				background-clip: text;
				text-align: center;
				animation: fadeInUp 1s ease-out;
				position: relative;
				z-index: 10;
			}
			
			.app-tagline {
				text-align: center;
				font-size: 1.1rem;
				color: #94a3b8;
				margin-bottom: 2rem;
				animation: fadeInUp 1s ease-out 0.3s both;
			}
			
			
			/* Simple file uploader styling */
			.stFileUploader {
				border: 2px dashed #6c63ff !important;
				border-radius: 20px !important;
				padding: 20px !important;
				background: rgba(255,255,255,0.08) !important;
				transition: all 0.3s ease !important;
			}
			
			.stFileUploader:hover {
				border-color: #00c2ff !important;
				background: rgba(255,255,255,0.12) !important;
			}
			
			.pred-card {
				border-radius: 20px; 
				padding: 24px; 
				background: rgba(255,255,255,0.1);
				box-shadow: 0 15px 35px rgba(0,0,0,0.3);
				border: 1px solid rgba(255,255,255,0.15);
				backdrop-filter: blur(10px);
				animation: fadeInUp 0.6s ease-out;
			}
			
			.confidence-circle {
				width: 80px;
				height: 80px;
				border-radius: 50%;
				background: conic-gradient(from 0deg, #6c63ff, #00c2ff, #6c63ff);
				display: flex;
				align-items: center;
				justify-content: center;
				margin: 0 auto 16px;
				position: relative;
				animation: pulse 2s ease-in-out infinite;
			}
			
			.confidence-circle::before {
				content: '';
				position: absolute;
				width: 60px;
				height: 60px;
				border-radius: 50%;
				background: rgba(15, 23, 42, 0.9);
			}
			
			.confidence-text {
				position: relative;
				z-index: 1;
				font-weight: bold;
				font-size: 1.2rem;
			}
			
			.prob-bar { 
				height: 12px; 
				border-radius: 10px; 
				background: rgba(255,255,255,0.2); 
				overflow: hidden; 
				margin: 8px 0;
			}
			.prob-fill { 
				height: 100%; 
				background: linear-gradient(90deg,#6c63ff,#00c2ff); 
				border-radius: 10px;
				transition: width 0.8s ease;
			}
			
			.loading-spinner {
				display: inline-block;
				width: 20px;
				height: 20px;
				border: 3px solid rgba(255,255,255,0.3);
				border-radius: 50%;
				border-top-color: #6c63ff;
				animation: spin 1s ease-in-out infinite;
			}
			
			@keyframes spin {
				to { transform: rotate(360deg); }
			}
			
			h1, h2, h3, h4, h5, h6, p, span, div { color: #e5e7eb !important; }
			
			/* Mobile Responsiveness */
			@media (max-width: 768px) {
				.app-title { font-size: 2rem; }
				.upload-box { padding: 24px; }
				.pred-card { padding: 20px; }
			}
			
			/* Hide any accidental empty text inputs Streamlit may render */
			input[type="text"][value=""] { display: none !important; }
			
			/* Container spacing */
			.block-container { padding-top: 0.5rem; }
			
			/* Enhanced button styles */
			.stButton > button {
				background: linear-gradient(45deg, #6c63ff, #00c2ff);
				border: none;
				border-radius: 12px;
				padding: 0.5rem 1.5rem;
				font-weight: 600;
				transition: all 0.3s ease;
			}
			
			.stButton > button:hover {
				transform: translateY(-2px);
				box-shadow: 0 8px 20px rgba(108, 99, 255, 0.4);
			}
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.markdown("<div class='app-title'>🫁 Pneumonia Detection</div>", unsafe_allow_html=True)
	st.markdown("<div class='app-tagline'>AI-Powered Chest X-ray Analysis for Early Detection</div>", unsafe_allow_html=True)
	st.markdown("<h4 style='margin:6px 0 12px 0; opacity:.9'>📁 Upload Chest X-ray</h4>", unsafe_allow_html=True)

	# Upload area with proper drag and drop
	st.markdown("""
	<div style="text-align: center; margin-bottom: 20px; opacity: 0.8;">
		<p>🖼️ <strong>Drag & drop</strong> a chest X-ray here or <strong>click to browse</strong></p>
		<p style="font-size: 0.9rem;">Supported formats: JPG, JPEG, PNG • Max size: 10MB</p>
	</div>
	""", unsafe_allow_html=True)
	
	uploaded_file = st.file_uploader(
		"Upload Chest X-ray Image", 
		type=["jpg", "jpeg", "png"],
		help="Upload a clear chest X-ray image for analysis"
	)
	
	# File validation
	if uploaded_file is not None:
		if validate_image_file(uploaded_file):
			# Show file info
			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("📁 File Name", uploaded_file.name)
			with col2:
				st.metric("📏 File Size", f"{uploaded_file.size / 1024:.1f} KB")
			with col3:
				st.metric("📋 File Type", uploaded_file.type.split('/')[-1].upper())
		else:
			st.stop()

	# Visual spacing between sections
	st.divider()

	if uploaded_file is not None:
		# Cache bytes for analysis trigger - ensure we get fresh bytes
		uploaded_file.seek(0)  # Reset file pointer
		file_bytes = uploaded_file.read()
		
		# Only reset analysis if this is a different file
		current_file_name = uploaded_file.name
		if "current_file" not in st.session_state or st.session_state["current_file"] != current_file_name:
			st.session_state["analyzed"] = False
			st.session_state["last_score"] = None
			st.session_state["current_file"] = current_file_name
		
		st.session_state["uploaded_bytes"] = file_bytes
		# Analyze button shows first
		if not st.session_state["analyzed"]:
			col1, col2, col3 = st.columns([1, 2, 1])
			with col2:
				if st.button("🔎 Analyze Image", type="primary", use_container_width=True):
					# Enhanced loading with progress
					progress_bar = st.progress(0)
					status_text = st.empty()
					
					try:
						status_text.text("🔄 Loading AI model...")
						progress_bar.progress(25)
						model = load_model()
						
						status_text.text("🖼️ Processing image...")
						progress_bar.progress(50)
						
						# Get fresh bytes and process
						image_bytes = st.session_state["uploaded_bytes"]
						if not image_bytes:
							st.error("❌ No image data found. Please upload an image.")
							st.stop()
						
						input_tensor = preprocess_image(image_bytes)
						
						if input_tensor is None:
							st.error("❌ Failed to process image. Please try a different file.")
							st.stop()
						
						status_text.text("🧠 Running AI analysis...")
						progress_bar.progress(75)
						
						# Ensure input tensor is valid
						if input_tensor.shape != (1, 150, 150, 3):
							st.error(f"❌ Invalid image shape: {input_tensor.shape}. Expected (1, 150, 150, 3)")
							st.stop()
						
						pred = model.predict(input_tensor, verbose=0)
						
						# Fix NumPy deprecation warning by properly extracting scalar
						if np.ndim(pred) == 2:
							score = float(pred[0, 0])
						else:
							score = float(pred.item())
						
						status_text.text("✅ Analysis complete!")
						progress_bar.progress(100)
						
						st.session_state["last_score"] = score
						st.session_state["analyzed"] = True
						
						# Clear progress indicators
						time.sleep(0.5)
						progress_bar.empty()
						status_text.empty()
						
					except Exception as e:
						st.error(f"❌ Analysis failed: {str(e)}")
						st.error(f"Error type: {type(e).__name__}")
						progress_bar.empty()
						status_text.empty()
					
					st.rerun()
		else:
			# Enhanced results display
			st.markdown("### 📊 Analysis Results")
			
			# Three-column layout for better organization
			col1, col2, col3 = st.columns([1, 1, 1])
			
			with col1:
				st.markdown("#### 🖼️ Image Preview")
				st.image(st.session_state["uploaded_bytes"], use_column_width=True)				
				# Clear button
				if st.button("🗑️ Clear Analysis", type="secondary"):
					st.session_state["analyzed"] = False
					st.session_state["uploaded_bytes"] = None
					st.session_state["last_score"] = None
					st.session_state["current_file"] = None
					st.rerun()
			
			with col2:
				st.markdown("#### 🤖 Model Information")
				st.info("**Model Type:** CNN (Convolutional Neural Network)\n\n**Input Size:** 150×150 pixels\n\n**Accuracy:** ~95% on test data")
			
			with col3:
				st.markdown("#### 🎯 Prediction Results")
				score = float(st.session_state["last_score"]) if st.session_state["last_score"] is not None else 0.0
				threshold = 0.50  # Fixed 50% threshold
				label, _probability = get_prediction_label_and_prob(score, threshold)
				conf_level, conf_strength = get_confidence_level(score, threshold)
				
				# Enhanced verdict card with circular progress
				if label == "Pneumonia":
					icon = "⚠️"
					bg_color = "rgba(220, 38, 38, 0.15)"
					border_color = "rgba(220, 38, 38, 0.5)"
					text_color = "#fca5a5"
				else:
					icon = "✅"
					bg_color = "rgba(16, 185, 129, 0.15)"
					border_color = "rgba(16, 185, 129, 0.5)"
					text_color = "#6ee7b7"
				
				st.markdown(f"<div class=\"pred-card\" style=\"background:{bg_color}; border:2px solid {border_color};\">", unsafe_allow_html=True)
				
				# Circular confidence indicator with percentage
				confidence_percentage = int(conf_strength * 100)
				st.markdown(f"""
				<div class="confidence-circle" style="background: conic-gradient(from 0deg, {border_color}, {border_color} {conf_strength*360}deg, rgba(255,255,255,0.2) {conf_strength*360}deg);">
					<div class="confidence-text" style="color: {text_color};">
						<div style="font-size: 1rem; font-weight: bold;">{conf_level}</div>
						<div style="font-size: 0.8rem; opacity: 0.8;">{confidence_percentage}%</div>
					</div>
				</div>
				""", unsafe_allow_html=True)
				
				st.markdown(f"<div style='font-size:2.5rem; font-weight:800; margin:0 0 12px 0; text-align:center; color: {text_color};'> {icon} <span>{label}</span></div>", unsafe_allow_html=True)
				st.markdown(f"<div style='text-align:center; margin-bottom:12px;'><b>Confidence Level:</b> <span style='color: {text_color}; font-weight:bold;'>{conf_level}</span></div>", unsafe_allow_html=True)
				
				# Enhanced progress bar
				st.markdown(f'<div class="prob-bar"><div class="prob-fill" style="width: {conf_strength*100:.0f}%; background: linear-gradient(90deg, {border_color}, {text_color});"></div></div>', unsafe_allow_html=True)
				
				
				# Risk assessment
				if label == "Pneumonia":
					st.warning("⚠️ **High Risk Detected**\n\nPlease consult a healthcare professional immediately for proper diagnosis and treatment.")
				else:
					st.success("✅ **Low Risk**\n\nNo signs of pneumonia detected. Continue monitoring symptoms.")
				
				st.markdown('</div>', unsafe_allow_html=True)

				# Grad-CAM checkbox and visualization
				gradcam_toggle = st.checkbox("🔍 Show Grad-CAM Heatmap", key="gradcam_checkbox", help="Visualize where the model is focusing on the X-ray.")
				if gradcam_toggle:
					model = load_model()
					input_tensor = preprocess_image(st.session_state["uploaded_bytes"])
					if input_tensor is not None:
						heatmap, _err = generate_gradcam(model, input_tensor, last_conv_layer_name=None)
						if heatmap is not None:
							gradcam_image, overlay_err = overlay_heatmap(st.session_state["uploaded_bytes"], heatmap, intensity=0.5)
							if gradcam_image is not None:
								st.image(gradcam_image, caption="Grad-CAM Heatmap: Model’s Focus Region", use_container_width=True)
							else:
								st.warning(overlay_err)
						else:
							st.warning(_err)
							# Help user see which conv layers are available
							conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
							if conv_layers:
								st.info(f"Available Conv2D layers: {conv_layers}")
							# Show full layer types for advanced troubleshooting
							all_layers = [(layer.name, type(layer).__name__) for layer in model.layers]
							st.write("All model layers:", all_layers)
					else:
						st.warning("Could not preprocess image for Grad-CAM generation.")

	# Additional spacing before further info
	st.divider()

	# Enhanced information section
	st.markdown("<h4 style='margin:20px 0 10px 0; opacity:.9'>📚 Educational Resources</h4>", unsafe_allow_html=True)

	# Master toggle button for full view with enhanced styling
	col1, col2, col3 = st.columns([1, 2, 1])
	with col2:
		btn_label = "📖 Close Full View" if st.session_state["expand_all"] else "📖 Open Full View"
		if st.button(btn_label, key="full_view_toggle", use_container_width=True):
			st.session_state["expand_all"] = not st.session_state["expand_all"]
			st.rerun()

	# Three equal-width expanders with enhanced icons and styling
	col_info, col_symptoms, col_precautions = st.columns([1, 1, 1])
	with col_info:
		with st.expander("🔬 What is Pneumonia?", expanded=st.session_state["expand_all"]):
			st.markdown("#### 🫁 What is Pneumonia?")
			st.markdown("""
			Pneumonia is an infection that inflames the air sacs (alveoli) in one or both lungs.
			The air sacs may fill with fluid or pus, making it difficult to breathe and causing
			cough, fever, and difficulty breathing.
			""")
			
			st.markdown("#### 🦠 How Does It Develop?")
			st.markdown("""
			Pneumonia occurs when germs (bacteria, viruses, or fungi) enter the lungs and overwhelm
			the immune system. These pathogens can be inhaled from the air or spread from other
			infections in the body.
			""")
			
			st.markdown("#### ⚠️ Risk Factors")
			st.markdown("""
			- Age (very young or elderly)
			- Weakened immune system
			- Chronic lung diseases
			- Smoking
			- Recent viral infections
			""")

	with col_symptoms:
		with st.expander("🩺 Symptoms & Signs", expanded=st.session_state["expand_all"]):
			st.markdown("#### 🌡️ Common Symptoms")
			st.markdown("""
			- **Cough:** May produce green/yellow or bloody mucus
			- **Fever:** High temperature with sweating and chills
			- **Breathing:** Shortness of breath or rapid breathing
			- **Chest Pain:** Sharp or stabbing pain, worse with deep breaths
			- **Fatigue:** Extreme tiredness and weakness
			""")
			
			st.markdown("#### 🚨 Emergency Signs")
			st.error("**Seek immediate medical attention if you experience:**")
			st.markdown("""
			- Difficulty breathing or rapid breathing
			- Confusion or changes in mental awareness
			- Bluish color of lips or fingernails
			- High fever (above 102°F/39°C)
			""")
   
			

	with col_precautions:
		with st.expander("🛡️ Treatment & Prevention", expanded=st.session_state["expand_all"]):
			st.markdown("#### 💊 Treatment")
			st.markdown("""
			- Follow your doctor's prescribed medications exactly
			- Get plenty of rest and avoid strenuous activity
			- Stay hydrated with water and clear fluids
			- Use humidifier or steam to ease breathing
			- Manage fever and pain as directed
			""")
			
			st.markdown("#### 🛡️ Prevention")
			st.success("**Protect yourself and others:**")
			st.markdown("""
			- Get vaccinated (flu and pneumococcal vaccines)
			- Practice good hand hygiene
			- Avoid smoking and secondhand smoke
			- Maintain a healthy lifestyle
			- Isolate when symptomatic
			""")

	st.divider()
	
	# Enhanced About section with additional features
	col1, col2 = st.columns([2, 1])
	
	with col1:
		st.markdown("### ℹ️ About This Application")
		
		# AI Technology section
		st.markdown("#### 🤖 AI Technology")
		st.info("This application uses a **Convolutional Neural Network (CNN)** trained on thousands of chest X-ray images to detect signs of pneumonia. The model achieves approximately **95% accuracy** on test data.")
		
		# How It Works section
		st.markdown("#### 🔬 How It Works")
		st.markdown("""
		- **Image Processing:** Uploaded images are resized to 150×150 pixels and normalized
		- **AI Analysis:** The CNN analyzes patterns and features in the X-ray
		- **Prediction:** Returns a probability score for pneumonia detection
		- **Threshold:** Adjustable sensitivity for different use cases
		""")
		
		# Disclaimer section
		st.markdown("#### ⚠️ Important Disclaimer")
		st.warning("**This tool is for educational and screening purposes only.** It should not replace professional medical diagnosis. Always consult with a qualified healthcare provider for proper medical evaluation.")
	
	with col2:
		st.markdown("### 🆘 Quick Help")
		
		# How to Use section
		st.markdown("#### 📋 How to Use")
		st.markdown("""
		1. Upload a clear chest X-ray image
		2. Click "Analyze Image"
		3. Review results and confidence level
		4. Consult healthcare provider if needed
		""")
		
		# Supported Formats section
		st.markdown("#### ✅ Supported Formats")
		st.markdown("""
		- JPG/JPEG
		- PNG
		- Max size: 10MB
		""")
		
		# Keyboard Shortcuts section
		st.markdown("#### ⌨️ Keyboard Shortcuts")
		st.markdown("""
		- **Enter** - Analyze image
		- **Esc** - Clear analysis
		""")
	
	# Footer with additional information
	st.markdown("---")
	st.markdown(
		"""
		<div style="text-align: center; opacity: 0.7; margin-top: 2rem;">
			<p>🫁 <strong>Pneumonia Detection App</strong> | Built with Streamlit & TensorFlow</p>
			<p style="font-size: 0.9rem;">For educational purposes only • Always consult healthcare professionals</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


if __name__ == "__main__":
	main()


