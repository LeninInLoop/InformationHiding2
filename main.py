import os
from typing import Dict, Tuple, List, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, floating
from scipy.fft import dctn, idctn
from PIL import Image
from skimage.util import view_as_blocks

from seedFinder import LowCorrelationSequenceGenerator

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
    def dctn(array: np.ndarray, axes: tuple, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return np.array(dctn(array, axes=axes, type=dct_type, norm=norm), dtype=np.float64)

    @staticmethod
    def idctn(array: np.ndarray, axes: tuple, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return np.array(idctn(array, axes=axes, type=dct_type, norm=norm), dtype=np.float64)

    @staticmethod
    def block_dct(image: np.ndarray, block_shape: int = 8) -> np.ndarray:
        blocks = view_as_blocks(image, block_shape=block_shape)
        return DCTUtils.dctn(blocks, axes=(2, 3))

    @staticmethod
    def block_idct(dct_blocks: np.ndarray) -> np.ndarray:
        recon_blocks = DCTUtils.idctn(dct_blocks, axes=(2, 3), norm='ortho')
        rows = [np.concatenate(row_blocks, axis=1) for row_blocks in recon_blocks]
        return np.concatenate(rows, axis=0)

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
    def generate_noise(length, n_streams=2, seed=None) -> np.ndarray:
        generator = LowCorrelationSequenceGenerator(
            master_seed = seed
            )
        if n_streams == 2:
            seeds, sequences, correlation = generator.find_low_correlation_seeds(
                equal_probability=False,
                target_correlation=0.003,
                max_attempts=20000,
                sequence_length=length
            )
        else:
            raise Exception("Only Implemented for 2 Sequence Generation")

        print(BColors.OK_GREEN + f"Best seeds found: {seeds}" + BColors.ENDC)
        print(BColors.OK_GREEN + f"Correlation: {correlation:.6f}" + BColors.ENDC)

        is_reproducible, regenerated_corr = generator.verify_reproducibility()
        print(BColors.OK_GREEN + f"Sequences are reproducible: {is_reproducible}" + BColors.ENDC)

        generator.save_seed_results("my_low_correlation_seeds.txt")
        return np.array(sequences, dtype=np.float32)

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
    def embed_watermark(dct_blocks, watermark, noise, gain_factor):
        result = dct_blocks.copy()
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()

        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                noise_pattern = noise[1] if watermark[i, j] else noise[0]
                print(f"\nBlock ({i},{j}) – watermark bit = {int(watermark[i,j])}")

                for idx, (u, v) in enumerate(high_freq_indices):
                    before = result[i, j, u, v]
                    delta  = gain_factor * noise_pattern[idx]

                    result[i, j, u, v] += delta
                    after  = result[i, j, u, v]
                    print(f"  HF[{u},{v}]: {before:.4f} + {delta:.4f} → {after:.4f}")

        return result

    @staticmethod
    def calculate_correlation(block_dct_coeffs, noise_patterns, high_freq_indices):
        high_freq_values = np.array([block_dct_coeffs[u, v] for (u, v) in high_freq_indices])
        print(f"\nExtracting from block – high-frequency coeffs:\n {high_freq_values}")

        correlations = np.zeros(2, dtype=np.float64)
        for bit in (0, 1):
            noise_values = noise_patterns[bit]
            corr = np.corrcoef(high_freq_values, noise_values)[0,1]
            correlations[bit] = corr
            print(f"  Corr with noise[{bit}]: {corr:.6f}")
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
            seed: int = 48,
    ) -> np.ndarray:
        """Embed watermark into image using the two-level DCT algorithm

        Algorithm steps as described in the paper:
        1. Compute BDCT of host image
        2. Create LRAI from DC coefficients
        3. Compute BDCT of LRAI
        4. Embed watermark as pseudo-random noise into high frequencies of LRAI's DCT
        5. Compute IBDCT of watermarked LRAI
        6. Replace DC coefficients of the original image with watermarked LRAI
        7. Compute IBDCT to get final watermarked image
        """
        print(f"{BColors.HEADER}Embedding watermark in image: {image_name}{BColors.ENDC}")

        # Step 1: Load and prepare the original image
        original_path = os.path.join(self.directories['image_base_path'], f"{image_name}.bmp")
        original_image = ImageUtils.load_image(original_path)

        print(f"{BColors.OK_BLUE}Original Image Size: {original_image.shape}{BColors.ENDC}")

        # Convert to grayscale if needed
        if len(original_image.shape) > 2:
            original_image = ImageUtils.convert_to_gray_scale(original_image)
            gray_image_path = os.path.join(self.directories["original_gray_path"], f"{image_name}_gray_scale.bmp")

            ImageUtils.save_image(path=gray_image_path, img=original_image)
            print(f"{BColors.OK_BLUE}Converted to Gray Scale: {original_image.shape}{BColors.ENDC}")

        # Resize to multiple of 8 if needed
        original_image = ImageUtils.resize_image(original_image, (512, 512))
        print(f"{BColors.OK_BLUE}Resized to : {original_image.shape}{BColors.ENDC}")

        # Step 1: Compute BDCT of host image
        original_dct_blocks = DCTUtils.block_dct(original_image, block_shape=(8,8))
        print(f"{BColors.OK_BLUE}Original DCT blocks shape: {original_dct_blocks.shape}{BColors.ENDC}")

        # Step 2: Create LRAI from DC coefficients
        dc_values = DCTUtils.extract_dc_values(original_dct_blocks)
        lrai_path = os.path.join(self.directories["low_res_approx_image_path"], f"{image_name}_lrai.bmp")
        ImageUtils.save_image(path=lrai_path, img=dc_values)


        print(f"{BColors.OK_BLUE}LRAI shape: {dc_values.shape}{BColors.ENDC}")

        # Step 3: Compute BDCT of LRAI
        lrai_dct_blocks = DCTUtils.block_dct(image=dc_values, block_shape=(8,8))

        print(f"{BColors.OK_BLUE}LRAI DCT blocks shape: {lrai_dct_blocks.shape}{BColors.ENDC}")

        # Load or create watermark
        watermark_path = os.path.join(self.directories["watermark_path"], f"watermark.bmp")
        watermark = WatermarkUtils.create_watermark_image(save_path=watermark_path, watermark_type=watermark_type)

        print(f"{BColors.OK_BLUE}Watermark Size: {watermark.shape}{BColors.ENDC}")

        # Generate noise patterns for bit 0 and bit 1
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()
        noise = WatermarkUtils.generate_noise(length=len(high_freq_indices), n_streams=2, seed=seed)

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
        watermarked_lrai = DCTUtils.block_idct(dct_blocks=watermarked_lrai_dct)

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
        watermarked_image = DCTUtils.block_idct(dct_blocks=watermarked_dct_blocks)

        # Save watermarked image
        watermarked_path = os.path.join(
            self.directories["watermarked_image_path"],
            f"Watermarked_{image_name}_gain_{gain_factor}.bmp"
        )
        ImageUtils.save_image(img=watermarked_image, path=watermarked_path)

        # Calculate PSNR
        mse = np.mean((original_image - watermarked_image) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

        print(f"{BColors.OK_GREEN}Watermarking completed successfully!{BColors.ENDC}")
        print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")

        return watermarked_image, watermarked_lrai_dct

    def extract_watermark(
            self,
            watermarked_image: np.ndarray,
            image_name: str,
            gain_factor: float = 30,
            seed: int = 48,
    ) -> tuple[ndarray[tuple[int, int], dtype[Any]], floating[Any]]:
        """Extract watermark from image using the two-level DCT algorithm

        Extraction algorithm as described in the paper:
        1. Do steps 1-3 from embedding algorithm (get LRAI's DCT blocks)
        2. Compute correlation between each block and both noise patterns
        3. Choose the pattern with higher correlation as the extracted bit
        """
        print(f"{BColors.HEADER}Extracting watermark from image: {image_name}{BColors.ENDC}")
        print(f"{BColors.OK_BLUE}Watermarked Image Size: {watermarked_image.shape}{BColors.ENDC}")

        # Step 1: Compute BDCT of watermarked image
        watermarked_dct_blocks = DCTUtils.block_dct(watermarked_image, block_shape=(8,8))

        # Extract DC values to create LRAI
        dc_values = DCTUtils.extract_dc_values(watermarked_dct_blocks)

        # Compute BDCT of LRAI
        lrai_dct_blocks = DCTUtils.block_dct(image=dc_values, block_shape=(8,8))

        print(f"{BColors.OK_BLUE}LRAI DCT blocks shape: {lrai_dct_blocks.shape}{BColors.ENDC}")

        # Load original watermark for comparison
        watermark_path = os.path.join(self.directories["watermark_path"], f"watermark.bmp")
        original_watermark = np.array(Image.open(watermark_path), dtype=np.bool_)
        watermark_height, watermark_width = original_watermark.shape

        # Generate the same noise patterns used for embedding
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()
        noise = WatermarkUtils.generate_noise(length=len(high_freq_indices), n_streams=2, seed=seed)

        # Step 2 & 3: Calculate correlation and extract watermark bits
        recovered_watermark = np.zeros((watermark_height, watermark_width), dtype=np.bool_)
        for i in range(lrai_dct_blocks.shape[0]):
            for j in range(lrai_dct_blocks.shape[1]):

                correlations = WatermarkUtils.calculate_correlation(
                    block_dct_coeffs=lrai_dct_blocks[i, j],
                    noise_patterns=noise,
                    high_freq_indices=high_freq_indices
                )

                # Choose A bit with higher correlation
                recovered_watermark[i, j] = np.argmax(correlations)

        # Save recovered watermark
        recovered_watermark_path = os.path.join(
            self.directories["extracted_watermark_path"],
            f"Recovered_Watermark_{image_name}_gain_{gain_factor}.bmp"
        )
        ImageUtils.save_image(img=recovered_watermark, path=recovered_watermark_path)

        # Calculate detection accuracy
        detection_accuracy = np.mean(recovered_watermark == original_watermark)

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


class Helper:
    @staticmethod
    def find_optimal_seed(
            watermarker,
            image_name,
            watermark_type=2,
            gain_factor=30,
            start_seed=0,
            max_seed=100,
            max_attempts=1000
    ):

        print(f"{BColors.HEADER}Finding optimal seed for image: {image_name}{BColors.ENDC}")
        print(f"{BColors.OK_BLUE}Parameters: watermark_type={watermark_type}, gain_factor={gain_factor}{BColors.ENDC}")
        print(f"{BColors.OK_BLUE}Testing seeds from {start_seed} to {max_seed}{BColors.ENDC}")

        best_seed = -1
        best_accuracy = 0.0
        attempts = 0

        # Load and prepare the original image
        original_path = os.path.join(watermarker.directories['image_base_path'], f"{image_name}.bmp")
        original_image = ImageUtils.load_image(original_path)

        # Convert to grayscale if needed
        if len(original_image.shape) > 2:
            original_image = ImageUtils.convert_to_gray_scale(original_image)

        # Resize to multiple of 8 if needed
        original_image = ImageUtils.resize_image(original_image, (512, 512))

        for seed in range(start_seed, max_seed + 1):
            attempts += 1
            if attempts > max_attempts:
                break

            print(f"\n{BColors.OK_CYAN}Testing seed: {seed} (Attempt {attempts}/{max_attempts}){BColors.ENDC}")

            try:
                # Embed watermark with current seed
                watermarked_image, _ = watermarker.embed_watermark(
                    image_name=image_name,
                    watermark_type=watermark_type,
                    gain_factor=gain_factor,
                    seed=seed,
                )

                # Extract watermark and check accuracy
                _, accuracy = watermarker.extract_watermark(
                    watermarked_image=watermarked_image,
                    image_name=image_name,
                    gain_factor=gain_factor,
                    seed=seed,
                )

                print(f"{BColors.OK_BLUE}Seed {seed} achieved accuracy: {accuracy * 100:.2f}%{BColors.ENDC}")

                # Update best seed if current is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_seed = seed
                    print(
                        f"{BColors.OK_GREEN}New best seed found: {best_seed} with accuracy {best_accuracy * 100:.2f}%{BColors.ENDC}")

                # If we reached 100% accuracy, break
                if accuracy == 1.0:
                    print(f"{BColors.OK_GREEN}Found perfect seed: {seed} with 100% accuracy!{BColors.ENDC}")
                    return seed

            except Exception as e:
                print(f"{BColors.FAIL}Error testing seed {seed}: {str(e)}{BColors.ENDC}")

        if best_seed != -1:
            print(
                f"\n{BColors.WARNING}Best seed found: {best_seed} with accuracy {best_accuracy * 100:.2f}%{BColors.ENDC}")
        else:
            print(f"\n{BColors.FAIL}No suitable seed found after {attempts} attempts{BColors.ENDC}")

        return best_seed

    @staticmethod
    def batch_test_seeds(watermarker, image_names, watermark_types, gain_factors, seed_ranges):
        start_seed, max_seed, max_attempts = seed_ranges
        results = {}

        for image_name in image_names:
            for watermark_type in watermark_types:
                for gain_factor in gain_factors:
                    print(
                        f"\n{BColors.HEADER}Testing configuration: Image={image_name}, Type={watermark_type}, Gain={gain_factor}{BColors.ENDC}")

                    optimal_seed = Helper.find_optimal_seed(
                        watermarker=watermarker,
                        image_name=image_name,
                        watermark_type=watermark_type,
                        gain_factor=gain_factor,
                        start_seed=start_seed,
                        max_seed=max_seed,
                        max_attempts=max_attempts
                    )

                    config_key = f"{image_name}_type{watermark_type}_gain{gain_factor}"
                    results[config_key] = optimal_seed

        # Print summary of results
        print(f"\n{BColors.HEADER}Summary of Optimal Seeds:{BColors.ENDC}")
        for config, seed in results.items():
            accuracy_text = "100%" if seed != -1 else "Failed"
            print(f"{BColors.OK_BLUE}{config}: Seed={seed}, Accuracy={accuracy_text}{BColors.ENDC}")

        return results

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
    watermarked_image, watermarked_lrai_dct = watermarker.embed_watermark(
        image_name="lenna",
        watermark_type=2,
        gain_factor=30,
        seed=2537, # GoldHill = 104, Lenna = 2537
    )

    # Extract and verify watermark
    recovered_watermark, accuracy = watermarker.extract_watermark(
        watermarked_image=watermarked_image,
        image_name="lenna",
        gain_factor=30,
        seed=2537,
    )

    print(f"Watermark detection accuracy: {accuracy * 100:.2f}%")


# Example of usage in main():
def main_with_seed_finder():
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

    # # Find optimal seed for a single image
    # optimal_seed = Helper.find_optimal_seed(
    #     watermarker=watermarker,
    #     image_name="lenna",
    #     watermark_type=2,
    #     gain_factor=30,
    #     start_seed=1,
    #     max_seed=10000,
    #     max_attempts=5000
    # )
    #
    # if optimal_seed != -1:
    #     print(f"Found optimal seed: {optimal_seed}")
    #
    #     # Use the optimal seed for final watermarking
    #     watermarked_image, watermarked_lrai_dct = watermarker.embed_watermark(
    #         image_name="lenna",
    #         watermark_type=2,
    #         gain_factor=30,
    #         seed=optimal_seed,
    #     )
    #
    #     # Extract and verify watermark
    #     recovered_watermark, accuracy = watermarker.extract_watermark(
    #         watermarked_image=watermarked_image,
    #         image_name="lenna",
    #         gain_factor=30,
    #         seed=optimal_seed,
    #     )
    #
    #     print(f"Final watermark detection accuracy: {accuracy * 100:.2f}%")
    # else:
    #     print("Failed to find an optimal seed with 100% accuracy.")

    # Optional: Test multiple configurations in batch
    results = Helper.batch_test_seeds(
        watermarker=watermarker,
        image_names=["lenna"],
        watermark_types=[2],
        gain_factors=[30],
        seed_ranges=(1, 100000, 5000)
    )

if __name__ == '__main__':
    # main_with_seed_finder()
    main()