import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Contoh parameter awal
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

# Dynamically update parameters based on selected method
def update_parameters(method):
    if method == "Brightness Adjustment":
        parameter['brightness_beta'] = st.slider("Intensitas Kecerahan", -50, 50, parameter.get('brightness_beta', 0), key="brightness_beta_slider")
    elif method == "Treshold":
        parameter['batas'] = st.slider("Ambang Batas", 0, 255, parameter.get('batas', 127), key="treshold_slider")
    elif method == "RGB":
        parameter["Red"] = st.slider("Red", 0, 255, parameter.get('Red', 127), key="red_slider")
        parameter["Green"] = st.slider("Green", 0, 255, parameter.get('Green', 127), key="green_slider")
        parameter["Blue"] = st.slider("Blue", 0, 255, parameter.get('Blue', 127), key="blue_slider")
    elif method == "Contrast Adjustment":
        parameter['contrast_threshold'] = st.slider("Nilai Ambang Kontras (m)", 0, 255, parameter.get('contrast_threshold', 127), key="contrast_threshold_slider")
        parameter['contrast_type'] = st.selectbox("Tipe Perbaikan Kontras", ["stretching", "thresholding"], index=["stretching", "thresholding"].index(parameter.get('contrast_type', "stretching")), key="contrast_type_selectbox")
    elif method == "Edge Detection":
        parameter['edge_detection_low'] = st.slider("Ambang Batas Rendah", 0, 255, parameter.get('edge_detection_low', 50), key="edge_detection_low_slider")
        parameter['edge_detection_high'] = st.slider("Ambang Batas Tinggi", 0, 255, parameter.get('edge_detection_high', 150), key="edge_detection_high_slider")
    elif method == "Rotation":
        parameter['rotation_angle'] = st.slider("Sudut Rotasi", -180, 180, parameter.get('rotation_angle', 45), key="rotation_angle_slider")
    elif method == "Mirroring":
        parameter['mirroring_type'] = st.selectbox("Jenis Mirroring", ["Horizontal", "Vertical", "Both"], index=["Horizontal", "Vertical", "Both"].index(parameter.get('mirroring_type', "Horizontal")), key="mirroring_type_selectbox")

# Define the Video Processor Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.category = None
        self.method = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.category and self.method:
            img_rgb = process_image(img_rgb, self.method)

        return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

# Image processing function
def process_image(image_array, method):
    if method == "Grayscale":
        grayscale_image = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
        return grayscale_image.astype(np.uint8)
    elif method == "Negative":
        return 255 - image_array
    elif method == "Brightness Adjustment":
        brightness_adjusted = np.clip(image_array + parameter['brightness_beta'], 0, 255)
        return brightness_adjusted.astype(np.uint8)
    elif method == "Edge Detection":
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, parameter['edge_detection_low'], parameter['edge_detection_high'])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif method == "Mirroring":
        if parameter['mirroring_type'] == "Horizontal":
            return image_array[:, ::-1]
        elif parameter['mirroring_type'] == "Vertical":
            return image_array[::-1]
        elif parameter['mirroring_type'] == "Both":
            return image_array[::-1, ::-1]
    elif method == "Rotation":
        height, width = image_array.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), parameter['rotation_angle'], 1)
        return cv2.warpAffine(image_array, matrix, (width, height))
    return image_array

# Upload image function
def upload_image():
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        category = st.selectbox("Pilih Kategori", ["Grayscale", "Negative", "Brightness Adjustment", "Edge Detection", "Mirroring", "Rotation"])
        method = category
        update_parameters(method)

        processed_image = process_image(image_array, method)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Gambar Asli", use_column_width=True)
        col2.image(processed_image, caption="Gambar Setelah Diolah", use_column_width=True)

# Display camera with WebRTC
def display_camera():
    category = st.selectbox("Pilih Kategori", ["Grayscale", "Negative", "Brightness Adjustment", "Edge Detection", "Mirroring", "Rotation"])
    method = category
    update_parameters(method)

    webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# Sidebar menu
menu = st.sidebar.radio("Pilih Opsi", ["Upload Gambar", "Tampilan Kamera"])
if menu == "Upload Gambar":
    st.subheader("Upload Gambar untuk Pengolahan")
    upload_image()
elif menu == "Tampilan Kamera":
    st.subheader("Tampilan Kamera Real-Time")
    display_camera()
