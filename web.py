import streamlit as st
import numpy as np
import cv2
import pywt
import scipy.linalg
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Set page configuration
st.set_page_config(page_title="Digital Watermarking System", layout="wide")

# Function for PPLU decomposition
def pplu_decomposition(A):
    P, L, U = scipy.linalg.lu(A)
    return P, L, U

# Function for Arnold Transform
def arnold_transform(image, iterations):
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)

    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                new_x = (2*x + y) % width
                new_y = (x + y) % height
                result[new_y, new_x] = image[y, x]
        image = result.copy()
    return result

# Function for Inverse Arnold Transform
def inverse_arnold_transform(image, iterations):
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)

    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                new_x = (x - y) % width
                new_y = (-x + 2*y) % height
                result[new_y, new_x] = image[y, x]
        image = result.copy()
    return result

def embed_watermark_dwt(cover_image, watermark_image, alpha=0.075, wavelet='haar'):
    # Convert images to grayscale
    cover_image_gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY) if len(cover_image.shape) > 2 else cover_image.copy()
    watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY) if len(watermark_image.shape) > 2 else watermark_image.copy()
    
    # Resize watermark to be compatible with LL subband size
    coeffs = pywt.dwt2(cover_image_gray / 255.0, wavelet)
    LL, (LH, HL, HH) = coeffs
    watermark_size = min(LL.shape[0], LL.shape[1], 256)  # Ensure it fits in LL subband
    watermark_gray = cv2.resize(watermark_gray, (watermark_size, watermark_size))
    
    # Normalize and decompose watermark
    watermark_norm = watermark_gray / 255.0
    P, L, U = pplu_decomposition(watermark_norm)
    LU = np.matmul(L, U)
    scrambled_data = arnold_transform(LU, iterations=4)
    
    # Ensure watermark fits in subbands
    max_LL = np.max(np.abs(LL))
    scaling_factor = alpha * max_LL
    
    # Resize scrambled data to match subband dimensions
    scrambled_LL = cv2.resize(scrambled_data, (LL.shape[1], LL.shape[0]))
    scrambled_HH = cv2.resize(scrambled_data, (HH.shape[1], HH.shape[0]))
    
    # Embed watermark
    LL_modified = LL + scaling_factor * scrambled_LL
    HH_modified = HH + scaling_factor * scrambled_HH
    
    # Reconstruct image
    watermarked_image = pywt.idwt2((LL_modified, (LH, HL, HH_modified)), wavelet)
    watermarked_image = np.clip(watermarked_image * 255, 0, 255).astype(np.uint8)
    
    return watermarked_image, P

# Function to embed watermark using SVD
def embed_watermark_svd(cover_image, watermark_image, alpha=0.075):
    cover_image_gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY) if len(cover_image.shape) > 2 else cover_image.copy()
    watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY) if len(watermark_image.shape) > 2 else watermark_image.copy()
    watermark_gray = cv2.resize(watermark_gray, (256, 256))

    cover_image_norm = cover_image_gray / 255.0
    watermark_norm = watermark_gray / 255.0

    # Perform SVD on cover image
    U, S, Vt = np.linalg.svd(cover_image_norm, full_matrices=False)

    # Embed watermark into singular values
    S_modified = S + alpha * watermark_norm.flatten()[:len(S)]

    # Reconstruct watermarked image
    watermarked_image = np.dot(U, np.dot(np.diag(S_modified), Vt))
    watermarked_image = np.clip(watermarked_image * 255, 0, 255).astype(np.uint8)

    return watermarked_image, (U, S, Vt)

# Function to extract watermark using DWT
def extract_watermark_dwt(watermarked_image, original_image, P, alpha=0.075, wavelet='haar', iterations=4):
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY) if len(watermarked_image.shape) > 2 else watermarked_image.copy()
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) > 2 else original_image.copy()

    watermarked_norm = watermarked_gray / 255.0
    original_norm = original_gray / 255.0

    coeffs_watermarked = pywt.dwt2(watermarked_norm, wavelet)
    LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked

    coeffs_original = pywt.dwt2(original_norm, wavelet)
    LL_o, (LH_o, HL_o, HH_o) = coeffs_original

    max_LL = np.max(np.abs(LL_o))
    scaling_factor = alpha * max_LL

    scrambled_watermark_LL = (LL_w - LL_o) / scaling_factor
    scrambled_watermark_HH = (HH_w - HH_o) / scaling_factor

    descrambled_LL = inverse_arnold_transform(scrambled_watermark_LL, iterations)
    descrambled_HH = inverse_arnold_transform(scrambled_watermark_HH, iterations)

    # Ensure dimensions match for matrix multiplication
    descrambled_LL = cv2.resize(descrambled_LL, (P.shape[1], P.shape[0]))
    descrambled_HH = cv2.resize(descrambled_HH, (P.shape[1], P.shape[0]))

    extracted_watermark_LL = np.matmul(P, descrambled_LL)
    extracted_watermark_HH = np.matmul(P, descrambled_HH)

    extracted_watermark_LL = (extracted_watermark_LL - np.min(extracted_watermark_LL)) / (np.max(extracted_watermark_LL) - np.min(extracted_watermark_LL))
    extracted_watermark_HH = (extracted_watermark_HH - np.min(extracted_watermark_HH)) / (np.max(extracted_watermark_HH) - np.min(extracted_watermark_HH))

    return extracted_watermark_LL, extracted_watermark_HH

# Function to extract watermark using SVD
def extract_watermark_svd(watermarked_image, original_image, svd_data, alpha=0.075):
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY) if len(watermarked_image.shape) > 2 else watermarked_image.copy()
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) > 2 else original_image.copy()

    watermarked_norm = watermarked_gray / 255.0
    original_norm = original_gray / 255.0

    U, S_original, Vt = svd_data

    # Perform SVD on watermarked image
    U_w, S_w, Vt_w = np.linalg.svd(watermarked_norm, full_matrices=False)

    # Extract watermark
    extracted_watermark = (S_w - S_original) / alpha

    # Dynamically calculate the largest possible square dimensions
    actual_size = extracted_watermark.size
    side_length = int(np.ceil(np.sqrt(actual_size)))

    # Pad the extracted watermark with zeros if necessary
    padded_watermark = np.zeros((side_length * side_length,))
    padded_watermark[:actual_size] = extracted_watermark

    extracted_watermark = padded_watermark.reshape(side_length, side_length)
    extracted_watermark = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))

    return extracted_watermark

# Attack functions
def jpeg_attack(image, quality=70):
    _, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    attacked_image = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
    return attacked_image

def noise_attack(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int32)
    attacked_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return attacked_image

def rescale_attack(image, scale_factor=0.5):
    h, w = image.shape[:2]
    resized_down = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    attacked_image = cv2.resize(resized_down, (w, h))
    return attacked_image

def blur_attack(image, kernel_size=5):
    attacked_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return attacked_image

def histogram_attack(image):
    attacked_image = cv2.equalizeHist(image)
    return attacked_image

def contrast_attack(image, alpha=1.5, beta=10):
    attacked_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return attacked_image

def gamma_attack(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    attacked_image = cv2.LUT(image, table)
    return attacked_image

def crop_attack(image, crop_ratio=0.25):
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    attacked_image = image.copy()
    attacked_image[0:crop_h, 0:crop_w] = 0
    return attacked_image

def rotation_attack(image, angle=45):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    attacked_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return attacked_image

def calculate_psnr(original, watermarked):
    # Convert to grayscale if needed
    if len(original.shape) > 2:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(watermarked.shape) > 2:
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    
    # Ensure uint8 type
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8)
    if watermarked.dtype != np.uint8:
        watermarked = (watermarked * 255).astype(np.uint8)
    
    return psnr(original, watermarked)

def calculate_cc(original_watermark, extracted_watermark):
    original_flat = original_watermark.flatten()
    extracted_flat = extracted_watermark.flatten()
    cc, _ = pearsonr(original_flat, extracted_flat)
    return cc

def main():
    st.title("Digital Image Watermarking System")
    st.write("This application demonstrates DWT and SVD based watermarking techniques with various attacks.")

    # Check for sample images
    if not os.path.exists('lena.png'):
        st.warning("Sample images not found. Please upload your own images.")

    # Sidebar controls
    st.sidebar.header("Watermarking Parameters")
    alpha = st.sidebar.slider("Alpha (embedding strength)", 0.01, 0.2, 0.075, 0.005)
    method = st.sidebar.selectbox("Watermarking Method", ["DWT", "SVD"])

    if method == "DWT":
        wavelet = st.sidebar.selectbox("Wavelet Type", ['haar', 'db1', 'db2', 'sym2', 'coif1'])
    else:
        wavelet = None

    attack_type = st.sidebar.selectbox("Select Attack Type", [
        "None", "JPEG Compression", "Noise", "Rescaling",
        "Blurring", "Histogram Equalization", "Contrast Adjustment",
        "Gamma Correction", "Cropping", "Rotation"
    ])

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        cover_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
    with col2:
        watermark_file = st.file_uploader("Upload Watermark Image", type=["png", "jpg", "jpeg"])

    if cover_file is not None and watermark_file is not None:
        # Read images
        cover_image = cv2.imdecode(np.frombuffer(cover_file.read(), np.uint8), cv2.IMREAD_COLOR)
        watermark_image = cv2.imdecode(np.frombuffer(watermark_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display original images
        st.subheader("Original Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cover_image, channels="BGR", caption="Cover Image")
        with col2:
            st.image(watermark_image, channels="BGR", caption="Watermark Image")

        # Embed watermark
        if method == "DWT":
            watermarked_image, P = embed_watermark_dwt(cover_image, watermark_image, alpha, wavelet)
            svd_data = None
        else:
            watermarked_image, svd_data = embed_watermark_svd(cover_image, watermark_image, alpha)
            P = None

        # Apply selected attack
        attacked_image = watermarked_image.copy()
        if attack_type != "None":
            if attack_type == "JPEG Compression":
                attacked_image = jpeg_attack(watermarked_image)
            elif attack_type == "Noise":
                attacked_image = noise_attack(watermarked_image)
            elif attack_type == "Rescaling":
                attacked_image = rescale_attack(watermarked_image)
            elif attack_type == "Blurring":
                attacked_image = blur_attack(watermarked_image)
            elif attack_type == "Histogram Equalization":
                attacked_image = histogram_attack(watermarked_image)
            elif attack_type == "Contrast Adjustment":
                attacked_image = contrast_attack(watermarked_image)
            elif attack_type == "Gamma Correction":
                attacked_image = gamma_attack(watermarked_image)
            elif attack_type == "Cropping":
                attacked_image = crop_attack(watermarked_image)
            elif attack_type == "Rotation":
                attacked_image = rotation_attack(watermarked_image)

        # Extract watermark
        if method == "DWT":
            extracted_LL, extracted_HH = extract_watermark_dwt(attacked_image, cover_image, P, alpha, wavelet)
            extracted_svd = None
        else:
            extracted_svd = extract_watermark_svd(attacked_image, cover_image, svd_data, alpha)
            extracted_LL, extracted_HH = None, None

        # Calculate metrics
        psnr_value = calculate_psnr(cover_image, watermarked_image)

        watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)
        watermark_gray = cv2.resize(watermark_gray, (256, 256))
        normalized_watermark = watermark_gray / 255.0

        if method == "DWT":
            cc_LL = calculate_cc(normalized_watermark, extracted_LL)
            cc_HH = calculate_cc(normalized_watermark, extracted_HH)
        else:
            # Resize extracted watermark to match original watermark dimensions
            extracted_svd_resized = cv2.resize(extracted_svd, (normalized_watermark.shape[1], normalized_watermark.shape[0]))
            cc_svd = calculate_cc(normalized_watermark, extracted_svd_resized)

        # Display results
        st.subheader("Watermarking Results")
    
        col1, col2 = st.columns(2)
        with col1:
            # Check if image is color or grayscale before displaying
            if len(watermarked_image.shape) == 3:
                st.image(watermarked_image, channels="BGR", caption=f"Watermarked Image (PSNR: {psnr_value:.2f} dB)")
            else:
                st.image(watermarked_image, caption=f"Watermarked Image (PSNR: {psnr_value:.2f} dB)")
        
        with col2:
            if attack_type != "None":
                if len(attacked_image.shape) == 3:
                    st.image(attacked_image, channels="BGR", caption=f"After {attack_type}")
                else:
                    st.image(attacked_image, caption=f"After {attack_type}")
    
        st.subheader("Extracted Watermark")

        if method == "DWT":
            col1, col2 = st.columns(2)
            with col1:
                st.image(extracted_LL, caption=f"Extracted from LL Band (CC: {cc_LL:.4f})", clamp=True)
            with col2:
                st.image(extracted_HH, caption=f"Extracted from HH Band (CC: {cc_HH:.4f})", clamp=True)
        else:
            st.image(extracted_svd_resized, caption=f"Extracted using SVD (CC: {cc_svd:.4f})", clamp=True)

        # Display metrics
        st.subheader("Performance Metrics")
        if method == "DWT":
            st.write(f"**Correlation Coefficients:** LL Band = {cc_LL:.4f}, HH Band = {cc_HH:.4f}")
        else:
            st.write(f"**Correlation Coefficient (SVD):** {cc_svd:.4f}")

        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 4))
        if method == "DWT":
            ax.bar(['LL Band', 'HH Band'], [cc_LL, cc_HH], color=['blue', 'orange'])
        else:
            ax.bar(['SVD'], [cc_svd], color='green')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Watermark Extraction Quality')
        st.pyplot(fig)

        # Download buttons
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            # Convert watermarked image to bytes
            _, encoded_img = cv2.imencode('.png', watermarked_image)
            st.download_button(
                label="Download Watermarked Image",
                data=BytesIO(encoded_img.tobytes()),
                file_name="watermarked.png",
                mime="image/png"
            )
        with col2:
            if method == "DWT":
                # Convert extracted watermark to bytes
                _, encoded_extracted = cv2.imencode('.png', (extracted_LL * 255).astype(np.uint8))
                st.download_button(
                    label="Download Extracted Watermark",
                    data=BytesIO(encoded_extracted.tobytes()),
                    file_name="extracted_watermark.png",
                    mime="image/png"
                )
            else:
                _, encoded_extracted = cv2.imencode('.png', (extracted_svd_resized * 255).astype(np.uint8))
                st.download_button(
                    label="Download Extracted Watermark",
                    data=BytesIO(encoded_extracted.tobytes()),
                    file_name="extracted_watermark.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()