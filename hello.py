import streamlit as st
import requests
import numpy as np
import base64
import time
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Image Detection App",
    page_icon="🔍",
    layout="wide"
)

# Constants and configuration - with fallback
try:
    API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    API_KEY = ""

if not API_KEY:
    API_KEY = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    if not API_KEY:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use this app.")

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4-vision-preview"

# Sidebar for app options
st.sidebar.title("Detection Settings")

detection_mode = st.sidebar.radio(
    "Choose Detection Mode:",
    ["Object Detection", "Scene Analysis", "Text Recognition"]
)

confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Function to encode image to base64
def encode_image(image):
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(np.uint8(image))
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to make API request to OpenAI
def analyze_image(image, mode="Object Detection"):
    if not API_KEY:
        st.error("Please provide an OpenAI API key to use this feature!")
        return None
    
    # Encode the image
    base64_image = encode_image(image)
    
    # Prepare prompt based on detection mode
    if mode == "Object Detection":
        prompt = "Detect all objects in this image. For each object, provide: 1) the name of the object, 2) a confidence score from 0 to 1, and 3) a brief description. Format as a JSON with a list of objects."
    elif mode == "Scene Analysis":
        prompt = "Analyze this scene. Describe: 1) the overall setting, 2) key elements in the scene, 3) the mood or atmosphere, and 4) any notable activities happening. Format as a JSON."
    else:  # Text Recognition
        prompt = "Extract all visible text from this image. Provide: 1) the text content, 2) the location in the image (top, middle, bottom, left, right, etc.), and 3) confidence level for each text element. Format as a JSON."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800
    }
    
    try:
        with st.spinner("Analyzing image..."):
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                st.error("Unexpected API response format.")
                return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

# Main app title
st.title("🔍 Image Detection App")
st.write("Detect objects, analyze scenes, or recognize text in images using your camera or uploaded files.")

# Create tabs for camera and upload options
tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload Image"])

with tab1:
    st.header("Camera Detection")
    
    # Add camera options
    camera_options = st.columns(2)
    with camera_options[0]:
        camera_input = st.camera_input("Take a picture")
    
    with camera_options[1]:
        if camera_input:
            st.image(camera_input, caption="Camera Input", use_column_width=True)
            
            # Process camera image
            if st.button("Analyze Camera Image"):
                if not API_KEY:
                    st.error("Please provide an OpenAI API key in the sidebar first.")
                else:
                    image = Image.open(camera_input)
                    results = analyze_image(image, detection_mode)
                    
                    if results:
                        st.subheader("Detection Results")
                        st.json(results)

with tab2:
    st.header("Image Upload Detection")
    
    # Add upload options
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process uploaded image
        if st.button("Analyze Uploaded Image"):
            if not API_KEY:
                st.error("Please provide an OpenAI API key in the sidebar first.")
            else:
                results = analyze_image(image, detection_mode)
                
                if results:
                    st.subheader("Detection Results")
                    st.json(results)
                    
                    # Try to display the results in a more user-friendly way
                    try:
                        import json
                        parsed_results = json.loads(results)
                        if isinstance(parsed_results, dict) and "objects" in parsed_results:
                            st.subheader("Detected Objects")
                            for i, obj in enumerate(parsed_results["objects"]):
                                if obj["confidence"] >= confidence_threshold:
                                    st.write(f"**{obj['name']}** (Confidence: {obj['confidence']:.2f})")
                                    st.write(obj.get("description", ""))
                                    st.write("---")
                    except:
                        st.write("Could not parse the structured results.")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses OpenAI's Vision model to analyze images.
    You can:
    - Take pictures with your camera
    - Upload existing images
    - Detect objects, analyze scenes, or recognize text
    
    You'll need to provide your own OpenAI API key with access to the GPT-4 Vision model.
    """
)

# Instructions for setting up secrets
if not API_KEY:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Setting up API Keys")
    st.sidebar.info(
        """
        For better security, you can set up your API key as a secret:
        
        1. Create `.streamlit/secrets.toml` in your app directory
        2. Add: `OPENAI_API_KEY = "your-key-here"`
        3. For Streamlit Cloud: Add the secret in the app dashboard
        """
    )
