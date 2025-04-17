import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pywt
import cv2
from scipy.linalg import lu
from numpy.linalg import norm
import math
import scipy

def preprocess_images(cover_path, watermark_path):
    """
    Preprocess cover and watermark images
    """
    # Load cover image
    cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    if cover_img.shape[0] != 512 or cover_img.shape[1] != 512:
        cover_img = cv2.resize(cover_img, (512, 512))
    
    # Load watermark image
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_img.shape[0] != 256 or watermark_img.shape[1] != 256:
        watermark_img = cv2.resize(watermark_img, (256, 256))
    
    # Normalize pixel values between 0-1
    cover_img = cover_img / 255.0
    watermark_img = watermark_img / 255.0
    
    return cover_img, watermark_img

def pplu_decomposition(W):
    """
    Performs PPLU decomposition of the watermark image
    Returns P, L, U matrices and the LU product
    """
    # Apply LU decomposition with partial pivoting
    P, L, U = lu(W)
    
    # Compute the LU product
    LU = np.dot(L, U)
    
    return P, L, U, LU
def arnold_transform(image, iterations):
    """
    Apply Arnold transform to scramble the image
    """
    n = image.shape[0]
    scrambled = np.zeros_like(image)
    img_temp = image.copy()
    
    for _ in range(iterations):
        for x in range(n):
            for y in range(n):
                # Arnold transform: (x,y) -> ((x+y) mod n, (x+2y) mod n)
                new_x = (x + y) % n
                new_y = (x + 2*y) % n
                scrambled[new_x, new_y] = img_temp[x, y]
        img_temp = scrambled.copy()
    
    return scrambled

def inverse_arnold_transform(image, iterations):
    """
    Apply inverse Arnold transform to recover the original image
    """
    n = image.shape[0]
    unscrambled = np.zeros_like(image)
    img_temp = image.copy()
    
    for _ in range(iterations):
        for x in range(n):
            for y in range(n):
                # Inverse Arnold transform: (x,y) -> ((2x-y) mod n, (-x+y) mod n)
                new_x = (2*x - y) % n
                new_y = (-x + y) % n
                unscrambled[new_x, new_y] = img_temp[x, y]
        img_temp = unscrambled.copy()
    
    return unscrambled

def apply_dwt(image):
    """
    Apply 1-level DWT to the cover image
    Returns the coefficients (LL, LH, HL, HH)
    """
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    return LL, LH, HL, HH
def embed_watermark(LL, LH, HL, HH, D_W, C=0.075):
    """
    Embed the scrambled watermark into LL and HH bands
    """
    # Calculate scaling factor
    alpha = (norm(LL) / np.max(LL)) * C
    
    # Embed watermark
    LL_embed = LL + alpha * D_W
    HH_embed = HH + alpha * D_W
    
    return LL_embed, LH, HL, HH_embed, alpha

def reconstruct_image(LL, LH, HL, HH):
    """
    Reconstruct the image using IDWT
    """
    coeffs = LL, (LH, HL, HH)
    return pywt.idwt2(coeffs, 'haar')
def extract_watermark(watermarked_LL, original_LL, alpha, HH_watermarked=None, HH_original=None):
    """
    Extract the watermark from LL and optionally HH bands
    """
    # Extract from LL band
    D_W1 = (watermarked_LL - original_LL) / alpha
    
    # Extract from HH band if provided
    D_W2 = None
    if HH_watermarked is not None and HH_original is not None:
        D_W2 = (HH_watermarked - HH_original) / alpha
    
    return D_W1, D_W2

def recover_watermark(D_W, P, iterations):
    """
    Recover the watermark by applying inverse Arnold transform and using the key P
    """
    # Apply inverse Arnold transform
    unscrambled_LU = inverse_arnold_transform(D_W, iterations)
    
    # Reconstruct watermark using P
    P_inv = np.linalg.inv(P)
    recovered_watermark = np.dot(P_inv, unscrambled_LU)
    
    # Normalize to [0,1] range
    recovered_watermark = (recovered_watermark - np.min(recovered_watermark)) / (np.max(recovered_watermark) - np.min(recovered_watermark))
    
    return recovered_watermark
def calculate_psnr(original, watermarked):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    """
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_cc(original, extracted):
    """
    Calculate correlation coefficient (CC)
    """
    # Flatten the arrays
    original_flat = original.flatten()
    extracted_flat = extracted.flatten()

    # Check for zero standard deviation
    if np.std(original_flat) == 0 or np.std(extracted_flat) == 0:
        return 0  # Return 0 correlation if one of the arrays is constant

    # Compute correlation coefficient
    return np.corrcoef(original_flat, extracted_flat)[0, 1]
def main(cover_path, watermark_path, arnold_iterations=4):
    """
    Main function to execute the watermarking pipeline
    """
    # 1. Preprocess inputs
    cover_img, watermark_img = preprocess_images(cover_path, watermark_path)
    
    # 2. PPLU decomposition of watermark
    P, L, U, LU = pplu_decomposition(watermark_img)
    
    # 3. Scramble LU using Arnold Transform
    D_W = arnold_transform(LU, arnold_iterations)
    
    # 4. Apply 1-Level DWT to cover image
    LL, LH, HL, HH = apply_dwt(cover_img)
    
    # 5. Embed watermark into LL & HH bands
    LL_embed, LH, HL, HH_embed, alpha = embed_watermark(LL, LH, HL, HH, D_W)
    
    # 6. Reconstruct watermarked image using IDWT
    watermarked_img = reconstruct_image(LL_embed, LH, HL, HH_embed)
    
    # Ensure values are within [0,1] range
    watermarked_img = np.clip(watermarked_img, 0, 1)
    
    # 7. Apply 1-Level DWT to watermarked image (for extraction)
    w_LL, w_LH, w_HL, w_HH = apply_dwt(watermarked_img)
    
    # 8. Extract watermark
    D_W1, D_W2 = extract_watermark(w_LL, LL, alpha, w_HH, HH)
    
    # 9-10. Recover watermark
    recovered_from_LL = recover_watermark(D_W1, P, arnold_iterations)
    recovered_from_HH = recover_watermark(D_W2, P, arnold_iterations)
    
    # Calculate PSNR and CC
    psnr_value = calculate_psnr(cover_img, watermarked_img)
    cc_value_LL = calculate_cc(watermark_img, recovered_from_LL)
    cc_value_HH = calculate_cc(watermark_img, recovered_from_HH)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.title('Original Cover Image')
    plt.imshow(cover_img, cmap='gray')
    
    plt.subplot(3, 3, 2)
    plt.title('Original Watermark')
    plt.imshow(watermark_img, cmap='gray')
    
    plt.subplot(3, 3, 3)
    plt.title('Scrambled Watermark (D_W)')
    plt.imshow(D_W, cmap='gray')
    
    plt.subplot(3, 3, 4)
    plt.title(f'Watermarked Image\nPSNR: {psnr_value:.2f}dB')
    plt.imshow(watermarked_img, cmap='gray')
    
    plt.subplot(3, 3, 5)
    plt.title(f'Extracted from LL\nCC: {cc_value_LL:.4f}')
    plt.imshow(recovered_from_LL, cmap='gray')
    
    plt.subplot(3, 3, 6)
    plt.title(f'Extracted from HH\nCC: {cc_value_HH:.4f}')
    plt.imshow(recovered_from_HH, cmap='gray')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'cover_img': cover_img,
        'watermark_img': watermark_img,
        'watermarked_img': watermarked_img,
        'recovered_from_LL': recovered_from_LL,
        'recovered_from_HH': recovered_from_HH,
        'psnr': psnr_value,
        'cc_LL': cc_value_LL,
        'cc_HH': cc_value_HH,
        'P': P,  # Save this as the security key
        'alpha': alpha,
        'arnold_iterations': arnold_iterations
    }
def apply_attacks(watermarked_img, attack_type):
    """
    Apply different attacks to the watermarked image
    """
    attacked_img = watermarked_img.copy()
    
    if attack_type == 'jpeg_compression':
        # Simulate JPEG compression by saving and loading
        temp_path = 'temp_compressed.jpg'
        plt.imsave(temp_path, watermarked_img, cmap='gray')
        attacked_img = plt.imread(temp_path)
        if attacked_img.ndim == 3:  # Convert RGB to grayscale if needed
            attacked_img = np.mean(attacked_img, axis=2)
    
    elif attack_type == 'gaussian_noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 0.05, watermarked_img.shape)
        attacked_img = watermarked_img + noise
        attacked_img = np.clip(attacked_img, 0, 1)
    
    elif attack_type == 'resizing':
        # Resize down and up again
        small = cv2.resize(watermarked_img, (watermarked_img.shape[0]//2, watermarked_img.shape[1]//2))
        attacked_img = cv2.resize(small, (watermarked_img.shape[0], watermarked_img.shape[1]))
    
    elif attack_type == 'cropping':
        # Crop 25% of the image and resize back
        h, w = watermarked_img.shape
        attacked_img = watermarked_img[h//4:3*h//4, w//4:3*w//4]
        attacked_img = cv2.resize(attacked_img, (h, w))
    
    elif attack_type == 'histogram_equalization':
        # Apply histogram equalization
        attacked_img = (watermarked_img * 255).astype(np.uint8)
        attacked_img = cv2.equalizeHist(attacked_img) / 255.0
    
    return attacked_img

def test_robustness(results, attack_types):
    """
    Test the robustness of watermarking against various attacks
    """
    attack_results = {}
    
    for attack in attack_types:
        # Apply attack
        attacked_img = apply_attacks(results['watermarked_img'], attack)
        
        # Extract watermark from attacked image
        a_LL, a_LH, a_HL, a_HH = apply_dwt(attacked_img)
        D_W1, D_W2 = extract_watermark(a_LL, results['cover_LL'], results['alpha'], a_HH, results['cover_HH'])
        
        # Recover watermark
        recovered_from_LL = recover_watermark(D_W1, results['P'], results['arnold_iterations'])
        recovered_from_HH = recover_watermark(D_W2, results['P'], results['arnold_iterations'])
        
        # Calculate metrics
        psnr_value = calculate_psnr(results['cover_img'], attacked_img)
        cc_value_LL = calculate_cc(results['watermark_img'], recovered_from_LL)
        cc_value_HH = calculate_cc(results['watermark_img'], recovered_from_HH)
        
        attack_results[attack] = {
            'attacked_img': attacked_img,
            'recovered_from_LL': recovered_from_LL,
            'recovered_from_HH': recovered_from_HH,
            'psnr': psnr_value,
            'cc_LL': cc_value_LL,
            'cc_HH': cc_value_HH
        }
    
    # Display results
    plt.figure(figsize=(15, len(attack_types) * 3))
    
    for i, attack in enumerate(attack_types):
        result = attack_results[attack]
        
        plt.subplot(len(attack_types), 3, i*3 + 1)
        plt.title(f'Attacked: {attack}\nPSNR: {result["psnr"]:.2f}dB')
        plt.imshow(result['attacked_img'], cmap='gray')
        
        plt.subplot(len(attack_types), 3, i*3 + 2)
        plt.title(f'Extracted from LL\nCC: {result["cc_LL"]:.4f}')
        plt.imshow(result['recovered_from_LL'], cmap='gray')
        
        plt.subplot(len(attack_types), 3, i*3 + 3)
        plt.title(f'Extracted from HH\nCC: {result["cc_HH"]:.4f}')
        plt.imshow(result['recovered_from_HH'], cmap='gray')
    
    plt.tight_layout()
    plt.show()
    
    return attack_results
def run_watermarking_pipeline(cover_path, watermark_path):
    """
    Run the complete watermarking pipeline with robustness testing
    """
    # Run the main watermarking process
    print("Starting watermarking process...")
    results = main(cover_path, watermark_path)
    
    # Save LL and HH for extraction
    results['cover_LL'], _, _, results['cover_HH'] = apply_dwt(results['cover_img'])
    
    # Test robustness against attacks
    print("Testing robustness against attacks...")
    attack_types = ['jpeg_compression', 'gaussian_noise', 'resizing', 'cropping', 'histogram_equalization']
    attack_results = test_robustness(results, attack_types)
    
    # Print summary of results
    print("\nWatermarking Results Summary:")
    print(f"PSNR of watermarked image: {results['psnr']:.2f} dB")
    print(f"CC of extracted watermark from LL: {results['cc_LL']:.4f}")
    print(f"CC of extracted watermark from HH: {results['cc_HH']:.4f}")
    
    print("\nRobustness Test Results:")
    for attack in attack_types:
        print(f"{attack.capitalize()}:")
        print(f"  PSNR: {attack_results[attack]['psnr']:.2f} dB")
        print(f"  CC (LL): {attack_results[attack]['cc_LL']:.4f}")
        print(f"  CC (HH): {attack_results[attack]['cc_HH']:.4f}")
    
    return results, attack_results
# Example usage
cover_path = 'lena.png'  # Path to cover image (e.g., Lena 512x512)
watermark_path = 'logo.png'  # Path to watermark image (e.g., DDNT logo 256x256)

results, attack_results = run_watermarking_pipeline(cover_path, watermark_path)
