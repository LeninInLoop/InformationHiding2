import os
from typing import Dict, Tuple, Optional, List, Union, Any
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, floating
from scipy.fft import dctn, idctn
from PIL import Image


class BColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        return np.array(Image.open(path), dtype=np.float32)

    @staticmethod
    def normalize(img: np.ndarray, dtype: str) -> np.ndarray:
        max_img = np.max(img)
        min_img = np.min(img)
        if max_img == min_img:
            normalized_image = np.full_like(img, 127.5)
        else:
            normalized_image = (img - min_img) / (max_img - min_img) * 255
        return normalized_image.astype(np.uint8 if dtype == 'uint8' else np.float32)

    @staticmethod
    def save_image(path: str, img: np.ndarray) -> None:
        if img.dtype != np.uint8 and img.dtype != np.bool:
            img = ImageUtils.normalize(img, dtype='uint8')
        Image.fromarray(img).save(path)

    @staticmethod
    def convert_to_gray_scale(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize(image, dtype='uint8')
        return np.array(Image.fromarray(image).convert('L'))

    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize(image, dtype='uint8')
        return np.array(Image.fromarray(image).resize(target_size))


class DCTUtils:
    @staticmethod
    def dctn(array: np.ndarray, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return dctn(array, type=dct_type, norm=norm)

    @staticmethod
    def idctn(array: np.ndarray, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return idctn(array, type=dct_type, norm=norm)

    @staticmethod
    def block_dct(image: np.ndarray, block_size: int = 8) -> np.ndarray:
        height, width = image.shape

        h_blocks = height // block_size
        w_blocks = width // block_size
        dct_blocks = np.zeros(
            (h_blocks, w_blocks, block_size, block_size),
            dtype=np.float32
        )
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = image[
                        i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size
                        ]
                dct_blocks[i, j] = DCTUtils.dctn(block)
        return dct_blocks

    @staticmethod
    def block_idct(dct_blocks: np.ndarray, block_size: int = 8) -> np.ndarray:
        h_blocks, w_blocks, _, _ = dct_blocks.shape
        reconstructed_image = np.zeros(
            (h_blocks * block_size, w_blocks * block_size),
            dtype=np.float32
        )
        for i in range(h_blocks):
            for j in range(w_blocks):
                idct_block = DCTUtils.idctn(dct_blocks[i, j])
                reconstructed_image[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size
                ] = idct_block
        return reconstructed_image

    @staticmethod
    def extract_dc_values(dct_blocks: np.ndarray) -> np.ndarray:
        return dct_blocks[:, :, 0, 0]

    @staticmethod
    def replace_dc_values(dct_blocks: np.ndarray, dc_values: np.ndarray) -> np.ndarray:
        result = dct_blocks.copy()
        h_blocks, w_blocks = dc_values.shape
        for i in range(h_blocks):
            for j in range(w_blocks):
                result[i, j, 0, 0] = dc_values[i, j]
        return result


class WatermarkUtils:
    @staticmethod
    def create_watermark_image(save_path: str, watermark_type: int = 1) -> np.ndarray:
        if watermark_type == 1:
            watermark_array = np.array([
                [0, 1, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 0, 1, 1, 0, 1, 0],
            ], dtype=np.bool_)
        else:
            watermark_array = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 0]
            ], dtype=np.bool_)
        ImageUtils.save_image(img=watermark_array, path=save_path)
        return watermark_array

    @staticmethod
    def generate_noise(shape, n_streams=2, seed=None) -> np.ndarray:
        seeds = [seed + i if seed is not None else None for i in range(n_streams)]
        noises = []
        choices = [-1, 0, 1]
        probs = [1 / 3, 1 / 3, 1 / 3]

        for i in range(n_streams):
            rng = np.random.default_rng(seeds[i])
            noise = rng.choice(choices, size=shape, p=probs)
            noises.append(noise)

        return np.array(noises, dtype=np.float32)

    @staticmethod
    def is_high_frequency(u: int, v: int, threshold: int = 5) -> bool:
        return u >= threshold or v >= threshold

    @staticmethod
    def get_high_frequency_indices(block_size: int = 8, threshold: int = 5) -> List[Tuple[int, int]]:
        indices = []
        for u in range(block_size):
            for v in range(block_size):
                if WatermarkUtils.is_high_frequency(u, v, threshold):
                    indices.append((u, v))
        return indices

    @staticmethod
    def embed_watermark(
            dct_blocks: np.ndarray,
            watermark: np.ndarray,
            noise: np.ndarray,
            gain_factor: float
    ) -> np.ndarray:
        result = dct_blocks.copy()

        watermark_height, watermark_width = watermark.shape
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()

        for i in range(min(dct_blocks.shape[0], watermark_height)):
            for j in range(min(dct_blocks.shape[1], watermark_width)):
                # Select noise pattern based on watermark bit (0 or 1)
                noise_pattern = noise[1 if watermark[i, j] else 0]
                noise_idx = 0

                # Apply noise to high frequency components
                for u, v in high_freq_indices:
                    if noise_idx < len(noise_pattern):
                        result[i, j, u, v] += gain_factor * noise_pattern[noise_idx]
                        noise_idx += 1
        return result

    @staticmethod
    def calculate_correlation(
            block_dct_coeffs: np.ndarray,
            noise_patterns: np.ndarray,
            high_freq_indices: List[Tuple[int, int]]
    ) -> np.ndarray:
        high_freq_values = np.array([block_dct_coeffs[u, v] for u, v in high_freq_indices])
        correlations = np.zeros(2)
        for bit in range(2):
            noise_values = noise_patterns[bit][:len(high_freq_indices)]
            corr_matrix = np.corrcoef(high_freq_values, noise_values)
            correlations[bit] = corr_matrix[0, 1]
        return correlations


class TwoLevelDCTWatermark:

    def __init__(self, directories: Dict[str, str]):
        self.directories = directories
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for directory in self.directories.values():
            os.makedirs(directory, exist_ok=True)

    def embed_watermark(
            self,
            image_name: str,
            watermark_type: int,
            gain_factor: float = 30,
            noise_seed: int = 48,
            verbose: bool = True
    ) -> np.ndarray:
        """Embed watermark into image using the two-level DCT algorithm

        Algorithm steps as described in the paper:
        1. Compute BDCT of host image
        2. Create LRAI from DC coefficients
        3. Compute BDCT of LRAI
        4. Embed watermark as pseudo-random noise into high frequencies of LRAI's DCT
        5. Compute IBDCT of watermarked LRAI
        6. Replace DC coefficients of original image with watermarked LRAI
        7. Compute IBDCT to get final watermarked image
        """
        if verbose:
            print(f"{BColors.HEADER}Embedding watermark in image: {image_name}{BColors.ENDC}")

        # Step 1: Load and prepare the original image
        original_path = os.path.join(self.directories['image_base_path'], f"{image_name}.bmp")
        original_image = ImageUtils.load_image(original_path)

        if verbose:
            print(f"{BColors.OK_BLUE}Original Image Size: {original_image.shape}{BColors.ENDC}")

        # Convert to grayscale if needed
        if len(original_image.shape) > 2:
            original_image = ImageUtils.convert_to_gray_scale(original_image)
            gray_image_path = os.path.join(self.directories["original_gray_path"], f"{image_name}_gray_scale.bmp")
            ImageUtils.save_image(path=gray_image_path, img=original_image)

            if verbose:
                print(f"{BColors.OK_BLUE}Converted to Gray Scale: {original_image.shape}{BColors.ENDC}")

        # Resize to multiple of 8 if needed
        original_image = ImageUtils.resize_image(original_image, (512,512))
        if verbose:
            print(f"{BColors.OK_BLUE}Resized to : {original_image.shape}{BColors.ENDC}")

        # Step 1: Compute BDCT of host image
        original_dct_blocks = DCTUtils.block_dct(original_image, block_size=8)

        if verbose:
            print(f"{BColors.OK_BLUE}Original DCT blocks shape: {original_dct_blocks.shape}{BColors.ENDC}")

        # Step 2: Create LRAI from DC coefficients
        dc_values = DCTUtils.extract_dc_values(original_dct_blocks)
        lrai_path = os.path.join(self.directories["low_res_approx_image_path"], f"{image_name}_lrai.bmp")
        ImageUtils.save_image(path=lrai_path, img=dc_values)

        if verbose:
            print(f"{BColors.OK_BLUE}LRAI shape: {dc_values.shape}{BColors.ENDC}")

        # Step 3: Compute BDCT of LRAI
        lrai_dct_blocks = DCTUtils.block_dct(image=dc_values, block_size=8)

        if verbose:
            print(f"{BColors.OK_BLUE}LRAI DCT blocks shape: {lrai_dct_blocks.shape}{BColors.ENDC}")

        # Load or create watermark
        watermark_path = os.path.join(self.directories["watermark_path"], f"watermark.bmp")
        watermark = WatermarkUtils.create_watermark_image(save_path=watermark_path, watermark_type=watermark_type)

        if verbose:
            print(f"{BColors.OK_BLUE}Watermark Size: {watermark.shape}{BColors.ENDC}")

        # Generate noise patterns for bit 0 and bit 1
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()
        noise = WatermarkUtils.generate_noise(shape=(len(high_freq_indices),), n_streams=2, seed=noise_seed)

        if verbose:
            print(f"{BColors.OK_BLUE}Noise Patterns Size: {noise.shape}{BColors.ENDC}")
            print(f"{BColors.OK_BLUE}High Frequency Coefficients: {len(high_freq_indices)}{BColors.ENDC}")

        # Step 4: Embed watermark in high frequency coefficients of LRAI's DCT blocks
        watermarked_lrai_dct = WatermarkUtils.embed_watermark(
            dct_blocks=lrai_dct_blocks,
            watermark=watermark,
            noise=noise,
            gain_factor=gain_factor
        )

        # Step 5: Compute IBDCT of watermarked LRAI
        watermarked_lrai = DCTUtils.block_idct(dct_blocks=watermarked_lrai_dct, block_size=8)

        # Save watermarked LRAI
        watermarked_lrai_path = os.path.join(
            self.directories["watermarked_low_res_image_path"],
            f"{image_name}_watermarked_lrai_gain_{gain_factor}.bmp"
        )
        ImageUtils.save_image(img=watermarked_lrai, path=watermarked_lrai_path)

        # Step 6: Replace DC coefficients of original image with watermarked LRAI
        watermarked_dct_blocks = DCTUtils.replace_dc_values(
            dct_blocks=original_dct_blocks,
            dc_values=watermarked_lrai
        )

        # Step 7: Compute IBDCT to get final watermarked image
        watermarked_image = DCTUtils.block_idct(dct_blocks=watermarked_dct_blocks, block_size=8)

        # Save watermarked image
        watermarked_path = os.path.join(
            self.directories["watermarked_image_path"],
            f"Watermarked_{image_name}_gain_{gain_factor}.bmp"
        )
        ImageUtils.save_image(img=watermarked_image, path=watermarked_path)

        # Calculate PSNR
        mse = np.mean((original_image - watermarked_image) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

        if verbose:
            print(f"{BColors.OK_GREEN}Watermarking completed successfully!{BColors.ENDC}")
            print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")

        return watermarked_image

    def extract_watermark(
            self,
            image_name: str,
            gain_factor: float = 30,
            noise_seed: int = 48,
            verbose: bool = True
    ) -> tuple[ndarray[tuple[int, int], dtype[Any]], floating[Any]]:
        """Extract watermark from image using the two-level DCT algorithm

        Extraction algorithm as described in the paper:
        1. Do steps 1-3 from embedding algorithm (get LRAI's DCT blocks)
        2. Compute correlation between each block and both noise patterns
        3. Choose the pattern with higher correlation as the extracted bit
        4. Compare average correlation with threshold to detect watermark presence
        """
        if verbose:
            print(f"{BColors.HEADER}Extracting watermark from image: {image_name}{BColors.ENDC}")

        # Load watermarked image
        watermarked_path = os.path.join(
            self.directories["watermarked_image_path"],
            f"Watermarked_{image_name}_gain_{gain_factor}.bmp"
        )
        watermarked_image = ImageUtils.load_image(watermarked_path)

        if verbose:
            print(f"{BColors.OK_BLUE}Watermarked Image Size: {watermarked_image.shape}{BColors.ENDC}")

        # Convert to grayscale if needed
        if len(watermarked_image.shape) > 2:
            watermarked_image = ImageUtils.convert_to_gray_scale(watermarked_image)

        # Step 1: Compute BDCT of watermarked image
        watermarked_dct_blocks = DCTUtils.block_dct(watermarked_image, block_size=8)

        # Extract DC values to create LRAI
        dc_values = DCTUtils.extract_dc_values(watermarked_dct_blocks)

        # Compute BDCT of LRAI
        lrai_dct_blocks = DCTUtils.block_dct(image=dc_values, block_size=8)

        if verbose:
            print(f"{BColors.OK_BLUE}LRAI DCT blocks shape: {lrai_dct_blocks.shape}{BColors.ENDC}")

        # Load original watermark for comparison
        watermark_path = os.path.join(self.directories["watermark_path"], f"watermark.bmp")
        original_watermark = np.array(Image.open(watermark_path), dtype=np.bool_)
        watermark_height, watermark_width = original_watermark.shape

        # Generate the same noise patterns used for embedding
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()
        noise = WatermarkUtils.generate_noise(shape=(len(high_freq_indices),), n_streams=2, seed=noise_seed)

        # Step 2 & 3: Calculate correlation and extract watermark bits
        recovered_watermark = np.zeros((watermark_height, watermark_width), dtype=np.bool_)
        correlation_values = np.zeros((watermark_height, watermark_width, 2))

        for i in range(min(lrai_dct_blocks.shape[0], watermark_height)):
            for j in range(min(lrai_dct_blocks.shape[1], watermark_width)):

                correlations = WatermarkUtils.calculate_correlation(
                    block_dct_coeffs=lrai_dct_blocks[i, j],
                    noise_patterns=noise,
                    high_freq_indices=high_freq_indices
                )
                correlation_values[i, j] = correlations

                # Choose bit with higher correlation
                recovered_watermark[i, j] = np.argmax(correlations) == 1

        # Save recovered watermark
        recovered_watermark_path = os.path.join(
            self.directories["extracted_watermark_path"],
            f"Recovered_Watermark_{image_name}_gain_{gain_factor}.bmp"
        )
        ImageUtils.save_image(img=recovered_watermark, path=recovered_watermark_path)

        # Calculate detection accuracy
        detection_accuracy = np.mean(recovered_watermark == original_watermark)

        if verbose:
            print(f"{BColors.OK_GREEN}Watermark Extraction completed.{BColors.ENDC}")
            print(f"{BColors.OK_GREEN}Detection Accuracy: {detection_accuracy * 100:.2f}%{BColors.ENDC}")

            # Visualize watermark comparison
            self._visualize_watermark_comparison(
                original=original_watermark,
                recovered=recovered_watermark,
                image_name=image_name,
                gain_factor=gain_factor
            )
        return recovered_watermark, detection_accuracy

    def _visualize_watermark_comparison(
            self,
            original: np.ndarray,
            recovered: np.ndarray,
            image_name: str,
            gain_factor: float
    ) -> None:
        """Visualize comparison between original and recovered watermark"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original watermark
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Watermark')
        axes[0].axis('off')

        # Plot recovered watermark
        axes[1].imshow(recovered, cmap='gray')
        axes[1].set_title('Recovered Watermark')
        axes[1].axis('off')

        # Plot difference
        difference = np.abs(original.astype(np.int8) - recovered.astype(np.int8))
        axes[2].imshow(difference, cmap='hot')
        axes[2].set_title('Difference (Errors)')
        axes[2].axis('off')

        # Add text with accuracy metrics
        accuracy = np.mean(original == recovered) * 100
        plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.2f}%', ha='center', fontsize=12)

        # Save figure
        comparison_path = os.path.join(
            self.directories["watermark_visualization_path"],
            f"Watermark_Comparison_{image_name}_gain_{gain_factor}.png"
        )
        plt.savefig(comparison_path)
        plt.close()


def main():
    # Define directory structure
    directories = {
        "image_base_path": "Images",
        "original_gray_path": "Images/Original_Gray_Scale",
        "low_res_approx_image_path": "Images/LRAI",
        "watermark_path": "Images/Watermark",
        "watermarked_image_path": "Images/Watermarked_Images",
        "watermarked_low_res_image_path": "Images/Watermarked_Images/LRAI",
        "extracted_watermark_path": "Images/Extracted_Watermarks",
        "watermark_visualization_path": "Images/Watermark_Visualizations",
    }

    # Initialize Two-Level DCT Watermarking
    watermarker = TwoLevelDCTWatermark(directories)

    # Process image with watermark
    watermarker.embed_watermark(
        image_name="lenna",
        watermark_type=1,
        gain_factor=30,
        noise_seed=48,
        verbose=True
    )

    # Extract and verify watermark
    recovered_watermark, accuracy = watermarker.extract_watermark(
        image_name="lenna",
        gain_factor=30,
        noise_seed=48,
        verbose=True
    )

    print(f"Watermark detection accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()