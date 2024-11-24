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
    def display_camera():
    """Displays real-time webcam feed with dynamic processing options and histograms."""
    # Initialize session state for camera
    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False  # Initialize camera state

    # Attempt to access the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("Tidak dapat membuka kamera. Pastikan kamera Anda terhubung.")
        return  # Exit the function if the camera cannot be accessed

    # Define layout with two columns
    col1, col2 = st.columns(2)
    stframe_original = col1.empty()
    stframe_hist_original = col1.empty()
    stframe_processed = col2.empty()
    stframe_hist_processed = col2.empty()

    # Main category selection
    category = st.selectbox("Pilih Kategori", list(categories.keys()), key="category_select")

    # Sub-method selection based on the main category
    method = st.selectbox("Pilih Metode Pengolahan", categories[category], key="method_select")

    # Update parameters dynamically based on the selected method
    update_parameters(method)

    # Button to start and stop the camera feed
    if st.button("Mulai Kamera" if not st.session_state['run_camera'] else "Stop Kamera"):
        st.session_state['run_camera'] = not st.session_state['run_camera']

    # Main loop for camera feed
    while st.session_state['run_camera']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal mendapatkan frame dari kamera.")
            break

        # Convert frame to RGB format for consistent processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame based on the selected method
        processed_frame = process_image(frame_rgb, method)

        # Display the original frame in the first column
        stframe_original.image(frame_rgb, caption="Gambar Asli (Real-Time)", channels="RGB", use_column_width=True)
        stframe_hist_original.pyplot(plot_histogram(frame_rgb, "Histogram Warna (Asli)"))

        # Ensure processed frame has RGB channels for consistency
        if len(processed_frame.shape) == 2:  # If grayscale, convert to RGB for display
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)

        # Display the processed frame in the second column
        stframe_processed.image(processed_frame, caption=f"Gambar Setelah Diolah ({method})", channels="RGB", use_column_width=True)
        stframe_hist_processed.pyplot(plot_histogram(processed_frame, "Histogram Warna (Setelah Diolah)"))

        # Adding a short delay to reduce CPU load
        time.sleep(0.1)

    # Release the camera when stopping the feed
    cap.release()
    cv2.destroyAllWindows()



# Sidebar menu
menu = st.sidebar.radio("Choose Option", ["Upload Image", "Use Camera"])
if menu == "Upload Image":
    st.subheader("Upload and Process Image")
    upload_image()
else:
    display_camera()
