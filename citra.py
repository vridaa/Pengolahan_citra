import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time

# Parameter awal
parameter = {
    'brightness_beta': 0,
    'batas': 127,
    'contrast_alpha': 1.0,
    'smoothing_kernel': 3,
    'edge_detection_low': 50,
    'edge_detection_high': 150,
    'mirroring_type': "Horizontal",
    'rotation_angle': 0
}

# Kategori utama dan metode
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
    elif method == "RGB":
        parameter["Red"] = st.slider("Red", 0, 255, parameter.get('Red', 127))
        parameter["Green"] = st.slider("Green", 0, 255, parameter.get('Green', 127))
        parameter["Blue"] = st.slider("Blue", 0, 255, parameter.get('Blue', 127))
    elif method == "Contrast Adjustment":
        parameter['contrast_alpha'] = st.slider("Contrast Alpha", 0.5, 3.0, parameter.get('contrast_alpha', 1.0), step=0.1)
    elif method == "Edge Detection":
        parameter['edge_detection_low'] = st.slider("Low Threshold", 0, 255, parameter.get('edge_detection_low', 50))
        parameter['edge_detection_high'] = st.slider("High Threshold", 0, 255, parameter.get('edge_detection_high', 150))
    elif method == "Rotation":
        parameter['rotation_angle'] = st.slider("Rotation Angle", -180, 180, parameter.get('rotation_angle', 0))
    elif method == "Mirroring":
        parameter['mirroring_type'] = st.selectbox("Mirroring Type", ["Horizontal", "Vertical", "Both"])
    elif method == "Smoothing":
        parameter['smoothing_kernel'] = st.slider("Kernel Size", 3, 15, parameter.get('smoothing_kernel', 3), step=2)

# Fungsi untuk memproses gambar
def process_image(image_array, method):
    if method == "Grayscale":
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    elif method == "Negative":
        return 255 - image_array
    elif method == "Brightness Adjustment":
        return np.clip(image_array + parameter['brightness_beta'], 0, 255).astype(np.uint8)
    elif method == "Treshold":
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(gray_image, parameter['batas'], 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
    elif method == "RGB":
        r, g, b = parameter["Red"], parameter["Green"], parameter["Blue"]
        adjusted = image_array.copy()
        adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] * (r / 127), 0, 255)
        adjusted[:, :, 1] = np.clip(adjusted[:, :, 1] * (g / 127), 0, 255)
        adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] * (b / 127), 0, 255)
        return adjusted.astype(np.uint8)
    elif method == "Edge Detection":
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, parameter['edge_detection_low'], parameter['edge_detection_high'])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif method == "Smoothing":
        kernel_size = parameter['smoothing_kernel']
        return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    elif method == "Mirroring":
        if parameter['mirroring_type'] == "Horizontal":
            return cv2.flip(image_array, 1)
        elif parameter['mirroring_type'] == "Vertical":
            return cv2.flip(image_array, 0)
        elif parameter['mirroring_type'] == "Both":
            return cv2.flip(image_array, -1)
    elif method == "Rotation":
        angle = parameter['rotation_angle']
        height, width = image_array.shape[:2]
        center = (width // 2, height // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image_array, rot_matrix, (width, height))
    return image_array

# Fungsi untuk menampilkan histogram
def plot_histogram(image, title):
    fig, ax = plt.subplots()
    if len(image.shape) == 2:  # Grayscale
        ax.hist(image.ravel(), bins=256, color='black')
    else:  # RGB
        for i, color in enumerate(['red', 'green', 'blue']):
            ax.hist(image[:, :, i].ravel(), bins=256, color=color, alpha=0.5)
    ax.set_title(title)
    return fig

# Video processor untuk WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.method = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.method:
            processed = process_image(img_rgb, self.method)
        else:
            processed = img_rgb

        return av.VideoFrame.from_ndarray(processed, format="rgb24")

# Fungsi untuk kamera dengan kategori
def display_camera():
    st.subheader("Real-Time Camera Feed with Processing")
    category = st.selectbox("Select Category", list(categories.keys()))
    method = st.selectbox("Select Processing Method", categories[category])

    update_parameters(method)

    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.method = method

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
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(processed_image, caption="Processed Image", use_column_width=True)

        col1.pyplot(plot_histogram(image_array, "Original Histogram"))
        col2.pyplot(plot_histogram(processed_image, "Processed Histogram"))

# Sidebar menu
menu = st.sidebar.radio("Choose Option", ["Upload Image", "Use Camera"])
if menu == "Upload Image":
    upload_image()
else:
    display_camera()
