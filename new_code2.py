import numpy as np
import cv2
import pywt
import scipy.linalg
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr

# Function for PPLU decomposition
def pplu_decomposition(A):
    # Using scipy.linalg.lu for PPLU decomposition
    P, L, U = scipy.linalg.lu(A)
    # P is the permutation matrix, L is lower triangular, U is upper triangular
    return P, L, U

# Function for Arnold Transform
def arnold_transform(image, iterations):
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)
    
    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                # Apply the Arnold Transform formula
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
                # Apply the Inverse Arnold Transform formula
                new_x = (x - y) % width
                new_y = (-x + 2*y) % height
                result[new_y, new_x] = image[y, x]
        image = result.copy()
    
    return result

# Function to embed watermark into cover image
def embed_watermark(cover_image, watermark_image, alpha=0.075):
    # Convert cover image to grayscale if it's color
    if len(cover_image.shape) > 2:
        cover_image_gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
    else:
        cover_image_gray = cover_image.copy()
    
    # Convert watermark to grayscale if it's color
    if len(watermark_image.shape) > 2:
        watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)
    else:
        watermark_gray = watermark_image.copy()
    
    # Resize watermark to be smaller than cover image
    watermark_gray = cv2.resize(watermark_gray, (256, 256))
    
    # Normalize images to [0, 1]
    cover_image_norm = cover_image_gray / 255.0
    watermark_norm = watermark_gray / 255.0
    
    # PPLU decomposition of watermark
    P, L, U = pplu_decomposition(watermark_norm)
    
    # Calculate LU product
    LU = np.matmul(L, U)
    
    # Apply Arnold Transform to LU (scrambling)
    iterations = 4  # As per paper example
    scrambled_data = arnold_transform(LU, iterations)
    
    # Apply wavelet transform to cover image
    coeffs = pywt.dwt2(cover_image_norm, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Calculate scaling factor as per paper
    max_LL = np.max(np.abs(LL))
    scaling_factor = alpha * max_LL
    
    # Embed scrambled watermark into LL and HH sub-bands
    LL_modified = LL + scaling_factor * scrambled_data[:LL.shape[0], :LL.shape[1]]
    HH_modified = HH + scaling_factor * scrambled_data[:HH.shape[0], :HH.shape[1]]
    
    # Inverse wavelet transform to get watermarked image
    watermarked_image = pywt.idwt2((LL_modified, (LH, HL, HH_modified)), 'haar')
    
    # Ensure values are in proper range and convert back to uint8
    watermarked_image = np.clip(watermarked_image * 255, 0, 255).astype(np.uint8)
    
    return watermarked_image, P  # Return the watermarked image and permutation matrix

# Function to extract watermark from watermarked image
def extract_watermark(watermarked_image, original_image, P, alpha=0.075, iterations=4):
    # Convert to grayscale if needed
    if len(watermarked_image.shape) > 2:
        watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    else:
        watermarked_gray = watermarked_image.copy()
    
    if len(original_image.shape) > 2:
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_image.copy()
    
    # Normalize images
    watermarked_norm = watermarked_gray / 255.0
    original_norm = original_gray / 255.0
    
    # Apply wavelet transform to both images
    coeffs_watermarked = pywt.dwt2(watermarked_norm, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked
    
    coeffs_original = pywt.dwt2(original_norm, 'haar')
    LL_o, (LH_o, HL_o, HH_o) = coeffs_original
    
    # Calculate scaling factor
    max_LL = np.max(np.abs(LL_o))
    scaling_factor = alpha * max_LL
    
    # Extract scrambled watermark from LL and HH
    scrambled_watermark_LL = (LL_w - LL_o) / scaling_factor
    scrambled_watermark_HH = (HH_w - HH_o) / scaling_factor
    
    # Apply inverse Arnold transform
    descrambled_LL = inverse_arnold_transform(scrambled_watermark_LL, iterations)
    descrambled_HH = inverse_arnold_transform(scrambled_watermark_HH, iterations)
    
    # Apply permutation matrix to get back original watermark
    extracted_watermark_LL = np.matmul(P, descrambled_LL)
    extracted_watermark_HH = np.matmul(P, descrambled_HH)
    
    # Normalize to [0, 1]
    extracted_watermark_LL = (extracted_watermark_LL - np.min(extracted_watermark_LL)) / (np.max(extracted_watermark_LL) - np.min(extracted_watermark_LL))
    extracted_watermark_HH = (extracted_watermark_HH - np.min(extracted_watermark_HH)) / (np.max(extracted_watermark_HH) - np.min(extracted_watermark_HH))
    
    return extracted_watermark_LL, extracted_watermark_HH

# Function to calculate PSNR
def calculate_psnr(original, watermarked):
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8)
    if watermarked.dtype != np.uint8:
        watermarked = (watermarked * 255).astype(np.uint8)
    return psnr(original, watermarked)

# Function to calculate Correlation Coefficient (CC)
def calculate_cc(original_watermark, extracted_watermark):
    # Flatten the arrays
    original_flat = original_watermark.flatten()
    extracted_flat = extracted_watermark.flatten()
    
    # Calculate correlation coefficient
    cc, _ = pearsonr(original_flat, extracted_flat)
    return cc

# Attack functions (same as before)
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

# Main testing function with enhanced visualization
def test_watermarking_system():
    # Load images
    cover_image = cv2.imread('lena.png')
    watermark_image = cv2.imread('logo-1.png')
    
    if cover_image is None:
        raise FileNotFoundError("Cover image 'lena.png' not found. Please provide the image.")
    if watermark_image is None:
        raise FileNotFoundError("Watermark image 'logo.png' not found. Please provide the image.")
    
    # Embed watermark
    alpha = 0.075  # Strength factor as per paper
    watermarked_image, P = embed_watermark(cover_image, watermark_image, alpha)
    
    # Calculate PSNR
    psnr_value = calculate_psnr(cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY), 
                               cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY) if len(watermarked_image.shape) > 2 else watermarked_image)
    print(f"PSNR of watermarked image: {psnr_value:.2f} dB")
    
    # Create dictionary of attack functions
    attacks = {
        "JPEG Compression": lambda img: jpeg_attack(img, quality=70),
        "Noise": lambda img: noise_attack(img, mean=0, sigma=10),
        "Rescaling": lambda img: rescale_attack(img, scale_factor=0.5),
        "Blurring": lambda img: blur_attack(img, kernel_size=5),
        "Histogram Equalization": lambda img: histogram_attack(img),
        "Contrast Adjustment": lambda img: contrast_attack(img, alpha=1.5, beta=10),
        "Gamma Correction": lambda img: gamma_attack(img, gamma=1.5),
        "Cropping (25%)": lambda img: crop_attack(img, crop_ratio=0.25),
        "Rotation (45°)": lambda img: rotation_attack(img, angle=45)
    }
    
    # Dictionary to store results
    results = {
        "cover_image": cover_image,
        "watermark_image": watermark_image,
        "watermarked_image": watermarked_image,
        "permutation_matrix": P,
        "psnr": psnr_value,
        "attacks": {}
    }
    
    # Test each attack
    for attack_name, attack_func in attacks.items():
        print(f"\nTesting attack: {attack_name}")
        
        # Apply attack
        attacked_image = attack_func(watermarked_image)
        
        # Extract watermark from attacked image
        extracted_LL, extracted_HH = extract_watermark(attacked_image, cover_image, P, alpha)
        
        # Calculate CC for both extracted watermarks
        watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY) if len(watermark_image.shape) > 2 else watermark_image
        watermark_gray = cv2.resize(watermark_gray, (256, 256))
        normalized_watermark = watermark_gray / 255.0
        cc_LL = calculate_cc(normalized_watermark, extracted_LL)
        cc_HH = calculate_cc(normalized_watermark, extracted_HH)
        
        # Store results
        results["attacks"][attack_name] = {
            "attacked_image": attacked_image,
            "extracted_LL": extracted_LL,
            "extracted_HH": extracted_HH,
            "cc_LL": cc_LL,
            "cc_HH": cc_HH
        }
        
        print(f"  CC from LL band: {cc_LL:.4f}")
        print(f"  CC from HH band: {cc_HH:.4f}")
    
    # Visualize results
    visualize_results(results)
    
    return results

def visualize_results(results):
    # Create a figure to show original, watermark and watermarked images
    plt.figure(figsize=(18, 6))
    
    # Original cover image
    plt.subplot(1, 3, 1)
    if len(results["cover_image"].shape) == 3:
        plt.imshow(cv2.cvtColor(results["cover_image"], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(results["cover_image"], cmap='gray')
    plt.title("Original Cover Image")
    plt.axis('off')
    
    # Original watermark
    plt.subplot(1, 3, 2)
    if len(results["watermark_image"].shape) == 3:
        plt.imshow(cv2.cvtColor(results["watermark_image"], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(results["watermark_image"], cmap='gray')
    plt.title("Original Watermark")
    plt.axis('off')
    
    # Watermarked image
    plt.subplot(1, 3, 3)
    if len(results["watermarked_image"].shape) == 3:
        plt.imshow(cv2.cvtColor(results["watermarked_image"], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(results["watermarked_image"], cmap='gray')
    plt.title(f"Watermarked Image\nPSNR: {results['psnr']:.2f} dB")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create figures for each attack
    attacks = list(results["attacks"].keys())
    n_attacks = len(attacks)
    
    for attack_name in attacks:
        attack_data = results["attacks"][attack_name]
        
        plt.figure(figsize=(18, 6))
        plt.suptitle(f"Attack: {attack_name}\nCC (LL): {attack_data['cc_LL']:.4f}, CC (HH): {attack_data['cc_HH']:.4f}", y=1.05)
        
        # Attacked image
        plt.subplot(1, 3, 1)
        if len(attack_data["attacked_image"].shape) == 3:
            plt.imshow(cv2.cvtColor(attack_data["attacked_image"], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(attack_data["attacked_image"], cmap='gray')
        plt.title("Attacked Image")
        plt.axis('off')
        
        # Extracted watermark from LL
        plt.subplot(1, 3, 2)
        plt.imshow(attack_data["extracted_LL"], cmap='gray')
        plt.title(f"Extracted from LL\nCC: {attack_data['cc_LL']:.4f}")
        plt.axis('off')
        
        # Extracted watermark from HH
        plt.subplot(1, 3, 3)
        plt.imshow(attack_data["extracted_HH"], cmap='gray')
        plt.title(f"Extracted from HH\nCC: {attack_data['cc_HH']:.4f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Create bar graph for CC values
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(attacks))
    width = 0.35
    
    plt.bar(x - width/2, [results["attacks"][attack]["cc_LL"] for attack in attacks], width, label='LL Band')
    plt.bar(x + width/2, [results["attacks"][attack]["cc_HH"] for attack in attacks], width, label='HH Band')
    
    plt.ylabel('Correlation Coefficient (CC)')
    plt.title('Robustness Against Various Attacks')
    plt.xticks(x, attacks, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the test
results = test_watermarking_system()