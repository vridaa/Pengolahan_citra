import streamlit as st
import cv2
import numpy as np
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
    elif method == "RGB":
        image_array[:, :, 0] = np.clip(image_array[:, :, 0] * (parameter.get("Red", 1.0)), 0, 255)
        image_array[:, :, 1] = np.clip(image_array[:, :, 1] * (parameter.get("Green", 1.0)), 0, 255)
        image_array[:, :, 2] = np.clip(image_array[:, :, 2] * (parameter.get("Blue", 1.0)), 0, 255)
        return image_array
    elif method == "Gaussian Noise":
        noise = np.random.normal(0, 25, image_array.shape).astype(np.uint8)
        return cv2.add(image_array, noise)
    elif method == "Brightness Adjustment":
        return np.clip(image_array + parameter['brightness_beta'], 0, 255).astype(np.uint8)
    elif method == "Contrast Adjustment":
        return cv2.convertScaleAbs(image_array, alpha=parameter['contrast_alpha'], beta=0)
    elif method == "Smoothing":
        kernel_size = parameter['smoothing_kernel']
        return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    elif method == "Sharpening":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image_array, -1, kernel)
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
    elif method == "Translasi":
        M = np.float32([[1, 0, parameter.get('translasi_m', 0)], [0, 1, parameter.get('translasi_n', 0)]])
        return cv2.warpAffine(image_array, M, (image_array.shape[1], image_array.shape[0]))
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

# Video Processor untuk WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.method = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Proses gambar
        processed_img = process_image(img_rgb, self.method) if self.method else img_rgb

        return av.VideoFrame.from_ndarray(processed_img, format="rgb24")

# Fungsi untuk kamera dengan kategori
def display_camera():
    st.subheader("Real-Time Camera Feed with Processing")
    category = st.selectbox("Select Category", list(categories.keys()))
    method = st.selectbox("Select Processing Method", categories[category])

    # Update parameter berdasarkan metode yang dipilih
    update_parameters(method)

    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.method = method

    # Menampilkan dua kolom: tampilan asli dan hasil
    col1, col2 = st.columns(2)
    if ctx.video_processor:
        col1.image(ctx.video_processor.recv, caption="Original Frame", channels="RGB", use_column_width=True)
        col2.image(ctx.video_processor.recv, caption="Processed Frame", channels="RGB", use_column_width=True)

        # Menampilkan histogram
        st.markdown("---")
        st.pyplot(plot_histogram(ctx.video_processor.recv, "Original Histogram"))
        st.pyplot(plot_histogram(ctx.video_processor.recv, "Processed Histogram"))

# Sidebar menu
menu = st.sidebar.radio("Choose Option", ["Upload Image", "Use Camera"])
if menu == "Upload Image":
    st.write("Upload Image functionality goes here.")
else:
    display_camera()
