import cv2
import numpy as np
from skimage.util import random_noise
from sklearn.cluster import KMeans
import os

# Function to apply quantization using k-means clustering
def apply_quantization(image_path, k=0, gaussian_blur_size=0, gaussian_blur_sigma=0,
                       gaussian_noise_std=0,salt_and_pepper_amount=0):
    # Load the image
    original_image = cv2.imread(image_path)

    # Convert the image to RGB (if it's in BGR format)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Apply Gaussian blur
    smoothed_image = cv2.GaussianBlur(original_image,(gaussian_blur_size, gaussian_blur_size),gaussian_blur_sigma)

    # Add Gaussian noise
    noisy_image = random_noise(smoothed_image, mode='gaussian', var=gaussian_noise_std**2)

    # Add salt-and-pepper noise
    noisy_image = random_noise(noisy_image, mode='s&p', amount=salt_and_pepper_amount)

    # Convert the image to a flat 1D array
    height, width, channels = noisy_image.shape
    flattened_image = noisy_image.reshape((height * width, channels))

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    quantized_colors = kmeans.fit_predict(flattened_image)

    # Create the quantized image
    quantized_image = kmeans.cluster_centers_[quantized_colors]
    quantized_image = quantized_image.reshape((height, width, channels))

    return quantized_image

def save_processing_image(quantized_image, output_folder, filename):
    output_path = os.path.join(output_folder, filename)

    # Scale the image to the range [0, 255]
    scaled_image = (quantized_image * 255).astype(np.uint8)

    # Save the image
    cv2.imwrite(output_path, cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR))
    print(f"Quantized image saved to {output_path}")

if __name__ == "__main__":
    input_folder = r'D:/Overlap 2D Shape 40 Percent'
    output_folder = r'D:/Overlap 2D Shape 40 Percent process_image_4'
    os.makedirs(output_folder, exist_ok=True)

    # Parameters for quantization
    k_value = 6
    gaussian_blur_size_value = 3
    gaussian_blur_sigma_value = 3
    gaussian_noise_std_value = 0.1
    poisson_noise_scale_value = 0.1
    salt_and_pepper_amount_value = 0.1

    # Get a list of image filenames in the input folder
    image_files = os.listdir(input_folder)

    for image_file in image_files:
        # Get the complete path for each image
        image_path = os.path.join(input_folder, image_file)

        # Apply quantization
        processing_image = apply_quantization(image_path, k_value, gaussian_blur_size_value,
                                             gaussian_blur_sigma_value, gaussian_noise_std_value,
                                             salt_and_pepper_amount_value)

        # Save the processing image
        output_filename = f"processing_{image_file}"
        save_processing_image(processing_image, output_folder, output_filename)