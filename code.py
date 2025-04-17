import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.ndimage
import pywt
from skimage import io, color, transform, metrics
def pplu_decomposition(A):
    """
    Perform PPLU decomposition on matrix A
    Returns: P, L, U matrices such that PA = LU
    """
    # Convert input to numpy array if not already
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Initialize matrices
    U = A.copy()
    P = np.eye(n)  # Identity matrix for permutation
    L = np.eye(n)  # Identity matrix for lower triangular

    for k in range(n-1):
        # Find pivot row for partial pivoting
        pivot_row = np.argmax(np.abs(U[k:, k])) + k

        # Swap rows in U and P
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            # For rows already calculated in L, also swap
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Calculate L and U values
        for i in range(k+1, n):
            if U[k, k] != 0:  # Avoid division by zero
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return P, L, U
def arnold_transform(image, iterations):
    """
    Apply Arnold Transform to an image for specified number of iterations
    """
    height, width = image.shape
    if height != width:
        raise ValueError("Image must be square for Arnold transform")

    result = image.copy()

    for _ in range(iterations):
        new_result = np.zeros_like(result)
        for x in range(height):
            for y in range(width):
                # Apply Arnold transform: [x', y'] = [1 1; 1 2] * [x, y] mod M
                new_x = (x + y) % height
                new_y = (x + 2*y) % width
                new_result[new_x, new_y] = result[x, y]
        result = new_result

    return result

def inverse_arnold_transform(image, iterations):
    """
    Apply inverse Arnold Transform to an image for specified number of iterations
    """
    height, width = image.shape
    if height != width:
        raise ValueError("Image must be square for Arnold transform")

    result = image.copy()

    for _ in range(iterations):
        new_result = np.zeros_like(result)
        for x in range(height):
            for y in range(width):
                # Apply inverse Arnold transform: [x', y'] = [2 -1; -1 1] * [x, y] mod M
                new_x = (2*x - y) % height
                new_y = (-x + y) % width
                new_result[new_x, new_y] = result[x, y]
        result = new_result

    return result
def embed_watermark(cover_image, watermark_image, alpha=0.075, at_iterations=4, wavelet='haar'):
    """
    Embed watermark into cover image using Hall property method

    Parameters:
    cover_image: Original image to embed watermark into
    watermark_image: Watermark to be embedded
    alpha: Strength factor (trade-off between imperceptibility and robustness)
    at_iterations: Number of Arnold Transform iterations
    wavelet: Wavelet transform type

    Returns:
    watermarked_image, P (permutation matrix for authentication)
    """
    # Ensure images are grayscale and normalized to [0,1]
    if len(cover_image.shape) > 2:
        cover_image = color.rgb2gray(cover_image)

    if len(watermark_image.shape) > 2:
        watermark_image = color.rgb2gray(watermark_image)

    # Normalize images to [0,1] range
    cover_image = cover_image / np.max(cover_image)
    watermark_image = watermark_image / np.max(watermark_image)

    # Resize watermark if necessary to be square
    size = min(watermark_image.shape)
    watermark_image = transform.resize(watermark_image, (size, size))

    # Apply PPLU decomposition to watermark
    P, L, U = pplu_decomposition(watermark_image)

    # Compute product of lower and upper triangular matrices
    LU_product = np.matmul(L, U)

    # Apply Arnold transform to product
    scrambled_data = arnold_transform(LU_product, at_iterations)

    # Apply wavelet transform to cover image
    coeffs = pywt.dwt2(cover_image, wavelet)
    LL, (LH, HL, HH) = coeffs

    # Calculate scaling factor
    scaling_factor = alpha * (np.max(np.abs(LL)))

    # Embed scrambled data into LL and HH sub-bands
    # Resize scrambled data if necessary
    scr_data_resized = transform.resize(scrambled_data, LL.shape)

    LL_modified = LL + scaling_factor * scr_data_resized
    HH_modified = HH + scaling_factor * scr_data_resized

    # Apply inverse wavelet transform
    watermarked_image = pywt.idwt2((LL_modified, (LH, HL, HH_modified)), wavelet)

    # Ensure output is in valid range [0,1]
    watermarked_image = np.clip(watermarked_image, 0, 1)

    return watermarked_image, P
def extract_watermark(watermarked_image, original_image, P, alpha=0.075, at_iterations=4, wavelet='haar'):
    """
    Extract watermark from watermarked image

    Parameters:
    watermarked_image: Image containing watermark
    original_image: Original cover image
    P: Permutation matrix (authentication key)
    alpha: Strength factor used in embedding
    at_iterations: Number of Arnold Transform iterations used
    wavelet: Wavelet transform type used

    Returns:
    extracted_watermark1, extracted_watermark2
    """
    # Ensure images are grayscale
    if len(watermarked_image.shape) > 2:
        watermarked_image = color.rgb2gray(watermarked_image)

    if len(original_image.shape) > 2:
        original_image = color.rgb2gray(original_image)

    # Normalize images to [0,1] range
    watermarked_image = watermarked_image / np.max(watermarked_image)
    original_image = original_image / np.max(original_image)

    # Apply wavelet transform to original and watermarked images
    coeffs_watermarked = pywt.dwt2(watermarked_image, wavelet)
    coeffs_original = pywt.dwt2(original_image, wavelet)

    LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked
    LL_o, (LH_o, HL_o, HH_o) = coeffs_original

    # Calculate scaling factor
    scaling_factor = alpha * (np.max(np.abs(LL_o)))

    # Extract scrambled data from LL and HH sub-bands
    extracted_scrambled1 = (LL_w - LL_o) / scaling_factor
    extracted_scrambled2 = (HH_w - HH_o) / scaling_factor

    # Apply inverse Arnold transform
    unscrambled1 = inverse_arnold_transform(extracted_scrambled1, at_iterations)
    unscrambled2 = inverse_arnold_transform(extracted_scrambled2, at_iterations)

    # Size of permutation matrix
    n = P.shape[0]

    # Resize unscrambled data to match permutation matrix size
    unscrambled1 = transform.resize(unscrambled1, (n, n))
    unscrambled2 = transform.resize(unscrambled2, (n, n))

    # Reconstruct watermark using permutation matrix
    # P·W = LU, so W = P^(-1)·LU
    P_inv = linalg.inv(P)
    extracted_watermark1 = np.matmul(P_inv, unscrambled1)
    extracted_watermark2 = np.matmul(P_inv, unscrambled2)

    # Ensure output is in valid range [0,1]
    extracted_watermark1 = np.clip(extracted_watermark1, 0, 1)
    extracted_watermark2 = np.clip(extracted_watermark2, 0, 1)

    return extracted_watermark1, extracted_watermark2

def calculate_psnr(original, watermarked):
    """Calculate Peak Signal-to-Noise Ratio"""
    return metrics.peak_signal_noise_ratio(original, watermarked)

def calculate_correlation_coefficient(watermark, extracted_watermark):
    """Calculate Correlation Coefficient between original and extracted watermark"""
    # Resize watermark to match extracted_watermark dimensions
    watermark_resized = transform.resize(watermark, extracted_watermark.shape)

    watermark_flat = watermark_resized.flatten() - np.mean(watermark_resized.flatten())
    extracted_flat = extracted_watermark.flatten() - np.mean(extracted_watermark.flatten())

    numerator = np.sum(watermark_flat * extracted_flat)
    denominator = np.sqrt(np.sum(watermark_flat**2) * np.sum(extracted_flat**2))

    if denominator == 0:
        return 0

    return numerator / denominator
# Complete example showing how to use the implementation

def main():
    # Load images
    cover_image = io.imread('lena.png')
    watermark_image = io.imread('logo.png')

    # Ensure cover_image has 3 channels (convert RGBA to RGB if necessary)
    if cover_image.shape[-1] == 4:  # Check if it has an alpha channel
        cover_image = color.rgba2rgb(cover_image)

    # Ensure watermark_image has 3 channels (convert RGBA to RGB if necessary)
    if watermark_image.shape[-1] == 4:  # Check if it has an alpha channel
        watermark_image = color.rgba2rgb(watermark_image)

    # Resize watermark_image to be square
    size = min(watermark_image.shape[:2])
    watermark_image = transform.resize(watermark_image, (size, size), anti_aliasing=True)

    # Resize cover_image to be square (if necessary)
    if cover_image.shape[0] != cover_image.shape[1]:
        size = min(cover_image.shape[:2])
        cover_image = transform.resize(cover_image, (size, size), anti_aliasing=True)

    # Parameters
    alpha = 0.075  # Embedding strength
    at_iterations = 4  # Arnold transform iterations
    wavelet_type = 'haar'

    # Embed watermark
    print("Embedding watermark...")
    watermarked_image, P = embed_watermark(
        cover_image,
        watermark_image,
        alpha=alpha,
        at_iterations=at_iterations,
        wavelet=wavelet_type
    )

    # Resize watermarked_image to match cover_image dimensions
    watermarked_image = transform.resize(watermarked_image, cover_image.shape, anti_aliasing=True)

    # Calculate PSNR
    psnr_value = calculate_psnr(
        cover_image / np.max(cover_image),
        watermarked_image
    )
    print(f"PSNR of watermarked image: {psnr_value:.2f} dB")

    # Simulate some attacks on watermarked image
    # 1. JPEG compression (simulated with blur)
    # Fix deprecated warning by using the correct import path
    attacked_image1 = scipy.ndimage.gaussian_filter(watermarked_image, sigma=0.5)

    # 2. Noise attack
    noise = np.random.normal(0, 0.01, watermarked_image.shape)
    attacked_image2 = np.clip(watermarked_image + noise, 0, 1)

    # Extract watermarks from original and attacked images
    print("Extracting watermarks...")
    extracted_wm1, extracted_wm2 = extract_watermark(
        watermarked_image,
        cover_image,
        P,
        alpha=alpha,
        at_iterations=at_iterations,
        wavelet=wavelet_type
    )

    extracted_wm1_attacked1, extracted_wm2_attacked1 = extract_watermark(
        attacked_image1,
        cover_image,
        P,
        alpha=alpha,
        at_iterations=at_iterations,
        wavelet=wavelet_type
    )

    extracted_wm1_attacked2, extracted_wm2_attacked2 = extract_watermark(
        attacked_image2,
        cover_image,
        P,
        alpha=alpha,
        at_iterations=at_iterations,
        wavelet=wavelet_type
    )

    # Calculate correlation coefficients
    # Convert watermark to grayscale if it's not already
    if len(watermark_image.shape) > 2:
        watermark_gray = color.rgb2gray(watermark_image)
    else:
        watermark_gray = watermark_image.copy()

    watermark_gray = watermark_gray / np.max(watermark_gray)

    cc_original = calculate_correlation_coefficient(
        watermark_gray,
        extracted_wm1
    )

    cc_attacked1 = calculate_correlation_coefficient(
        watermark_gray,
        extracted_wm1_attacked1
    )

    cc_attacked2 = calculate_correlation_coefficient(
        watermark_gray,
        extracted_wm1_attacked2
    )

    print(f"Correlation coefficient (no attack): {cc_original:.4f}")
    print(f"Correlation coefficient (blur attack): {cc_attacked1:.4f}")
    print(f"Correlation coefficient (noise attack): {cc_attacked2:.4f}")

    # Display results
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 3, 1)
    plt.imshow(cover_image, cmap='gray')
    plt.title('Original Cover Image')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(watermark_image, cmap='gray')
    plt.title('Original Watermark')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(watermarked_image, cmap='gray')
    plt.title(f'Watermarked Image\nPSNR: {psnr_value:.2f} dB')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(extracted_wm1, cmap='gray')
    plt.title(f'Extracted Watermark (LL)\nCC: {cc_original:.4f}')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(attacked_image1, cmap='gray')
    plt.title('Attacked Image (Blur)')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(extracted_wm1_attacked1, cmap='gray')
    plt.title(f'Extracted from Blur Attack\nCC: {cc_attacked1:.4f}')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(extracted_wm2, cmap='gray')
    plt.title('Extracted Watermark (HH)')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(attacked_image2, cmap='gray')
    plt.title('Attacked Image (Noise)')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(extracted_wm1_attacked2, cmap='gray')
    plt.title(f'Extracted from Noise Attack\nCC: {cc_attacked2:.4f}')
    plt.axis('off')

    plt.tight_layout()

    # Save the figure instead of showing it interactively
    plt.savefig('watermarking_results.png')
    plt.close()

    print("Process completed. Results saved to 'watermarking_results.png'")

if __name__ == "__main__":
    main()