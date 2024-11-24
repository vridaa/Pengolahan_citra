import streamlit as st
import numpy as np
from PIL import Image
import cv2  # Tambahkan OpenCV untuk akses kamera
import matplotlib.pyplot as plt
import uuid
from scipy.ndimage import affine_transform

# Judul aplikasi
st.title("Aplikasi Pengolahan Citra Sederhana")

# Menu utama dengan pilihan "Upload Gambar" atau "Tampilan Kamera"
menu = st.sidebar.radio("Pilih Opsi", ("Upload Gambar", "Tampilan Kamera"))

# Daftar metode pengolahan citra
methods = [
    "Grayscale", "Thresholding", "Negative", "Brightness Adjustment", "Mirroring", "Rotation"
]
def rotate_image(image_array, angle):
    # Convert the angle to radians
    theta = np.radians(angle)
    
    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Calculate the center of the image to keep it centered during rotation
    center_y, center_x = np.array(image_array.shape[:2]) / 2
    
    # Create the transformation matrix for centering, rotating, and translating back
    transform_matrix = np.array([
        [np.cos(theta), -np.sin(theta), center_x - center_x * np.cos(theta) + center_y * np.sin(theta)],
        [np.sin(theta),  np.cos(theta), center_y - center_x * np.sin(theta) - center_y * np.cos(theta)]
    ])
    
    # Apply the affine transformation with bilinear interpolation
    rotated_image = affine_transform(
        image_array,
        transform_matrix[:2, :2],
        offset=transform_matrix[:2, 2],
        order=1,  # Bilinear interpolation
        mode='nearest'  # Fill mode for out-of-bounds pixels
    )
    
    return rotated_image
def process_image(image_array, method):
    if method == "Grayscale":
        return np.mean(image_array, axis=2).astype(np.uint8)
    if method == "Thresholding":
        # Generate a unique key for the slider using uuid
        threshold_value = st.slider(
            "Nilai Threshold", 
            0, 255, 127, 
            key=f"{method}_threshold_slider_{uuid.uuid4()}"
        )
        
        gray_image = np.mean(image_array, axis=2).astype(np.uint8)
        thresholded_image = np.where(gray_image > threshold_value, 255, 0).astype(np.uint8)
        
        return thresholded_image

    elif method == "Negative":
        return 255 - image_array
    elif method == "Brightness Adjustment":
        beta = st.slider("Intensitas Kecerahan", -100, 100, 0, key=f"{method}_brightness_slider")
        return np.clip(image_array + beta, 0, 255).astype(np.uint8)
    elif method == "Mirroring":
        flip_type = st.selectbox("Jenis Mirroring", ["Horizontal", "Vertical", "Both"], key=f"{method}_mirroring_type")
        if flip_type == "Horizontal":
            return image_array[:, ::-1]
        elif flip_type == "Vertical":
            return image_array[::-1]
        elif flip_type == "Both":
            return image_array[::-1, ::-1]
    elif method == "Rotation":
        angle = st.slider("Sudut Rotasi (dalam derajat)", 0, 360, 0, step=1, key=f"{method}_rotation_slider")
        return rotate_image(image_array, angle)
    # Add other processing methods as needed
    else:
        return image_array

def plot_histogram(image_array, title):
    fig, ax = plt.subplots()
    
    # Pastikan citra memiliki 3 saluran warna (RGB)
    if len(image_array.shape) == 3:  # Citra berwarna
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image_array], [i], None, [256], [0, 256])
            ax.plot(histr, color=col)
    elif len(image_array.shape) == 2:  # Citra grayscale
        histr = cv2.calcHist([image_array], [0], None, [256], [0, 256])
        ax.plot(histr, color='k')  # Histogram grayscale
    ax.set_xlim([0, 256])
    ax.set_title(title)
    return fig



# Fungsi untuk upload gambar
def upload_image():
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))

        method = st.selectbox("Pilih Metode Pengolahan", methods)
        processed_image = process_image(image, method)

        col1, col2 = st.columns(2)
        
        col1.image(image, caption="Gambar Asli", use_column_width=True)
        col1.pyplot(plot_histogram(image, "Histogram Warna (Asli)"))

        col2.image(processed_image, caption=f"Gambar Setelah Diolah ({method})", use_column_width=True)
        col2.pyplot(plot_histogram(processed_image, "Histogram Warna (Setelah Diolah)"))

# Fungsi untuk tampilan kamera real-time
def display_camera():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.warning("Tidak dapat membuka kamera.")
        return

    # Set up columns
    col1, col2 = st.columns(2)

    # Placeholder for images and histograms
    stframe_original = col1.empty()
    stframe_hist_original = col1.empty()
    stframe_processed = col2.empty()
    stframe_hist_processed = col2.empty()

    # Select processing method
    method = st.selectbox("Pilih Metode Pengolahan", methods)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal mendapatkan frame dari kamera.")
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_image(frame_rgb, method)
        
        # Display original frame and its histogram in the first column
        stframe_original.image(frame_rgb, caption="Gambar Asli (Real-Time)", channels="RGB", use_column_width=True)
        stframe_hist_original.pyplot(plot_histogram(frame_rgb, "Histogram Warna (Asli)"))

        # Ensure processed frame has RGB channels for uniformity
        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)

        # Display processed frame and its histogram in the second column
        stframe_processed.image(processed_frame, caption=f"Gambar Setelah Diolah ({method})", channels="RGB", use_column_width=True)
        stframe_hist_processed.pyplot(plot_histogram(processed_frame, "Histogram Warna (Setelah Diolah)"))

    cap.release()
    cv2.destroyAllWindows()

if menu == "Upload Gambar":
    st.subheader("Upload Gambar untuk Pengolahan")
    upload_image()
elif menu == "Tampilan Kamera":
    st.subheader("Tampilan Kamera Real-Time")
    display_camera()
