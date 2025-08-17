
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Page Configuration and Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Image Processor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        color: #4CAF50;
        background-color: #FFFFFF;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #45a049;
        color: #FFFFFF;
        background-color: #45a049;
    }
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------

def process_image(image, operation, params):

    if operation == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif operation == 'Blur':
        ksize = params.get('ksize', 5)
        # Kernel size must be odd
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif operation == 'Edge Detection (Canny)':
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        t1 = params.get('threshold1', 100)
        t2 = params.get('threshold2', 200)
        return cv2.Canny(gray_img, t1, t2)
    elif operation == 'Thresholding':
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh_val = params.get('threshold_val', 127)
        _, processed = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
        return processed
    elif operation == 'Sepia':
        sepia_kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        # Apply the kernel and clip values to be in the 0-255 range
        sepia_img = cv2.transform(image, sepia_kernel)
        sepia_img = np.clip(sepia_img, 0, 255)
        return sepia_img.astype(np.uint8)
    return image

def create_histogram(image):

    fig = go.Figure()
    if len(image.shape) == 2:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        fig.add_trace(go.Bar(x=np.arange(256), y=hist.ravel(), name='Intensity', marker_color='#34495e'))
        fig.update_layout(title_text='Image Intensity Histogram', xaxis_title='Intensity Level', yaxis_title='Pixel Count')
    else:  # Color image
        colors = ('b', 'g', 'r')
        channel_names = ('Blue', 'Green', 'Red')
        plot_colors = ('#3498db', '#2ecc71', '#e74c3c')
        for i, (col, name, plot_color) in enumerate(zip(colors, channel_names, plot_colors)):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            fig.add_trace(go.Bar(x=np.arange(256), y=hist.ravel(), name=name, marker_color=plot_color))
        fig.update_layout(title_text='Color Histogram', xaxis_title='Intensity Level', yaxis_title='Pixel Count', barmode='overlay')
        fig.update_traces(opacity=0.75)
        
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#2c3e50'
    )
    return fig

# -------------------------------------------------------------------
# Main Application UI
# -------------------------------------------------------------------

def main():
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## Image Processor")
        st.markdown("---")
        st.markdown("### 1. Select Image Source")
        
        source_options = ["Upload an Image", "Use Webcam", "Image from URL"]
        source_choice = st.radio("Choose your input method:", source_options, label_visibility="collapsed")

        image_file = None
        if source_choice == "Upload an Image":
            image_file = st.file_uploader("Select an image file", type=["jpg", "jpeg", "png"])
        elif source_choice == "Use Webcam":
            image_file = st.camera_input("Smile! You're on camera.")
        elif source_choice == "Image from URL":
            url = st.text_input("Enter Image URL:", "")
            if url:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status() # Raise an exception for bad status codes
                    image_file = BytesIO(response.content)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching image from URL: {e}")
                    image_file = None

        st.markdown("---")
        
        # --- Processing Controls ---
        st.markdown("### 2. Configure Processing")
        
        processing_options = ['None', 'Grayscale', 'Blur', 'Edge Detection (Canny)', 'Thresholding', 'Sepia']
        operation = st.selectbox("Select an operation:", processing_options)
        
        params = {}
        if operation == 'Blur':
            params['ksize'] = st.slider("Blur Kernel Size", 1, 31, 5, 2)
        elif operation == 'Edge Detection (Canny)':
            params['threshold1'] = st.slider("Lower Threshold", 0, 255, 100)
            params['threshold2'] = st.slider("Upper Threshold", 0, 255, 200)
        elif operation == 'Thresholding':
            params['threshold_val'] = st.slider("Threshold Value", 0, 255, 127)


    # --- Main Content ---
    st.title("Image Processing Dashboard")
    st.markdown("Upload an image and apply various processing effects using the controls on the left.")
    
    if image_file:
        try:
            # Load and display original image
            original_pil = Image.open(image_file).convert("RGB")
            original_cv = np.array(original_pil)
            
            # Process the image
            if operation != 'None':
                processed_cv = process_image(original_cv, operation, params)
            else:
                processed_cv = original_cv
            
            # --- Display Images ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_pil, use_container_width=True)

            with col2:
                st.subheader("Processed Image")
                st.image(processed_cv, use_container_width=True)

            # --- Display Histogram ---
            st.markdown("---")
            st.subheader("Image Properties Graph")
            
            # Create histogram for the processed image
            histogram_fig = create_histogram(processed_cv)
            st.plotly_chart(histogram_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
            st.warning("Please try uploading a different image or checking the URL.")

    else:
        st.info("Please select an image source from the sidebar to get started.")

if __name__ == "__main__":
    main()
