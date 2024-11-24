import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Parameter awal
parameter = {
    'brightness_beta': 0,
    'batas': 127,
    'contrast_alpha': 1.0,
    'smoothing_kernel': 3,
    'edge_detection_low': 50,
    'edge_detection_high': 150,
    'mirroring_type': "Horizontal",
    'rotation_angle': 0,
}

# Kategori filter
categories = {
    "Filtering": ["Grayscale", "Negative", "Treshold", "RGB"],
    "Noise": ["Gaussian Noise"],
    "Adjustment": ["Brightness Adjustment", "Contrast Adjustment", "Smoothing", "Sharpening"],
    "Edge Detection": ["Edge Detection"],
    "Mirroring & Rotate": ["Mirroring", "Rotation"],
    "Translasi": ["Translasi"]
}

# Fungsi untuk memperbarui parameter
def update_parameters(method):
    if method == "Brightness Adjustment":
        parameter['brightness_beta'] = st.slider("Brightness Intensity", -50, 50, parameter.get('brightness_beta', 0))
    elif method == "Treshold":
        parameter['batas'] = st.slider("Threshold Value", 0, 255, parameter.get('batas', 127))
    elif method == "Edge Detection":
        parameter['edge_detection_low'] = st.slider("Low Threshold", 0, 255, parameter.get('edge_detection_low', 50))
        parameter['edge_detection_high'] = st.slider("High Threshold", 0, 255, parameter.get('edge_detection_high', 150))
    elif method == "Mirroring":
        parameter['mirroring_type'] = st.selectbox("Mirroring Type", ["Horizontal", "Vertical", "Both"], 
                                                   index=["Horizontal", "Vertical", "Both"].index(parameter.get('mirroring_type', "Horizontal")))
    elif method == "Rotation":
        parameter['rotation_angle'] = st.slider("Rotation Angle", -180, 180, parameter.get('rotation_angle', 0))
    elif method == "Gaussian Noise":
        parameter['prob_noise'] = st.slider("Gaussian Noise Probability", 0.0, 1.0, 0.05)

# Fungsi untuk memproses gambar
def process_image(image_array, method):
    if method == "Grayscale":
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    elif method == "Negative":
        return 255 - image_array
    elif method == "Treshold":
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        return cv2.threshold(gray, parameter['batas'], 255, cv2.THRESH_BINARY)[1]
    elif method == "Gaussian Noise":
        noise = np.random.normal(0, 25, image_array.shape).astype(np.uint8)
        return cv2.add(image_array, noise)
    elif method == "Brightness Adjustment":
        return np.clip(image_array + parameter['brightness_beta'], 0, 255).astype(np.uint8)
    elif method == "Edge Detection":
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, parameter['edge_detection_low'], parameter['edge_detection_high'])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif method == "Mirroring":
        if parameter['mirroring_type'] == "Horizontal":
            return image_array[:, ::-1]
        elif parameter['mirroring_type'] == "Vertical":
            return image_array[::-1, :]
        elif parameter['mirroring_type'] == "Both":
            return image_array[::-1, ::-1]
    elif method == "Rotation":
        (h, w) = image_array.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, parameter['rotation_angle'], 1.0)
        return cv2.warpAffine(image_array, M, (w, h))
    return image_array

# Fungsi untuk menampilkan histogram
def plot_histogram(image, title="Histogram"):
    """Plots the histogram of an image, handling both grayscale and RGB images."""
    fig, ax = plt.subplots()
    
    # Check if the image has multiple color channels (RGB) or is grayscale
    if len(image.shape) == 3:  # RGB image
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(histr, color=col)
    else:  # Grayscale image
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(histr, color='black')  # Use black for grayscale histogram
    
    ax.set_title(title)
    return fig

# Fungsi untuk upload gambar
def upload_image():
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        category = st.selectbox("Select Category", list(categories.keys()))
        method = st.selectbox("Select Processing Method", categories[category])

        update_parameters(method)
        processed_image = process_image(image_array, method)

        col1, col2 = st.columns(2)
        col1.image(image_array, caption="Original Image", use_column_width=True)
        col2.image(processed_image, caption="Processed Image", use_column_width=True)

        st.markdown("---")
        col1.pyplot(plot_histogram(image_array, "Original Histogram"))
        col2.pyplot(plot_histogram(processed_image, "Processed Histogram"))

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.method = None

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = process_image(img_rgb, self.method) if self.method else img_rgb
            return av.VideoFrame.from_ndarray(processed_img, format="rgb24")
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            return frame


def display_camera():
    """Displays real-time webcam feed with processing and histograms."""
    st.subheader("Real-Time Camera Feed with Processing")

    # Select category and processing method
    category = st.selectbox("Select Category", list(categories.keys()))
    method = st.selectbox("Select Processing Method", categories[category])

    # Update parameters dynamically
    update_parameters(method)

    # Create columns for original feed, processed feed, and histograms
    col1, col2 = st.columns(2)
    st.markdown("---")  # Add separator for histograms
    hist_col1, hist_col2 = st.columns(2)

    # WebRTC streamer

    ctx = webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
)


    if ctx.video_processor:
        # Assign selected method to VideoProcessor
        ctx.video_processor.method = method

        # Continuously process frames for real-time display
        while ctx.state.playing:
            try:
                # Access the original frame
                frame = ctx.video_processor.recv().to_ndarray(format="bgr24")
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame using the selected method
                processed_frame = process_image(original_frame, method)

                # Display original and processed frames
                col1.image(original_frame, caption="Original Frame", channels="RGB", use_column_width=True)
                col2.image(processed_frame, caption="Processed Frame", channels="RGB", use_column_width=True)

                # Display histograms for original and processed frames
                hist_col1.pyplot(plot_histogram(original_frame, "Original Frame Histogram"))
                hist_col2.pyplot(plot_histogram(processed_frame, "Processed Frame Histogram"))

                # Sleep to avoid overwhelming the app
                time.sleep(0.1)
            except Exception as e:
                st.error(f"Error in processing frame: {e}")
                break


# Sidebar menu
menu = st.sidebar.radio("Choose Option", ["Upload Image", "Use Camera"])
if menu == "Upload Image":
    st.subheader("Upload and Process Image")
    upload_image()
else:
    display_camera()
