from typing import Dict, Any
import numpy as np, os

from src.helpers import BColors, Visualization
from src.utils import DCTUtils, WatermarkUtils, ImageUtils

class TwoLevelDCTWatermarkExtraction:

    def __init__(self, directories: Dict[str, str]):
        self.directories = directories
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for directory in self.directories.values():
            os.makedirs(directory, exist_ok=True)

    def extract_watermark(
            self,
            watermarked_image: np.ndarray,
            image_name: str,
            gain_factor: float = 30,
            seed: int = 48,
            watermark_type: int = 1,
            equal_probability: bool = False,
            number_of_sequences: int = 2
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[Any]], np.floating[Any]]:
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
        watermark_path = os.path.join(
            self.directories["watermark_path"],
            f"watermark_type{watermark_type}_{number_of_sequences}.bmp"
        )
        original_watermark = ImageUtils.load_image(watermark_path).astype(np.bool)

        # Generate the same noise patterns used for embedding
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()
        noise = WatermarkUtils.generate_noise(
            length=len(high_freq_indices),
            n_streams=number_of_sequences,
            seed=seed,
            image_name=image_name,
            watermark_type=watermark_type,
            equal_probability=equal_probability,
        )

        # Step 2 & 3: Calculate correlation and extract watermark bits
        recovered_watermark = np.zeros((8, 8), dtype=np.uint8)  # Added dtype
        for i in range(lrai_dct_blocks.shape[0]):
            for j in range(lrai_dct_blocks.shape[1]):
                correlations = WatermarkUtils.calculate_correlation(
                    block_dct_coeffs=lrai_dct_blocks[i, j],
                    noise_patterns=noise,
                    high_freq_indices=high_freq_indices
                )
                recovered_watermark[i, j] = np.argmax(correlations)

        print(recovered_watermark)
        if number_of_sequences == 2:
            size = (8, 8)
        elif number_of_sequences == 4:
            size = (8, 16)
        elif number_of_sequences == 8:
            size = (12, 16)
        elif number_of_sequences == 16:
            size = (16, 16)
        else:
            raise NotImplementedError
        if number_of_sequences != 2:
            recovered_watermark = WatermarkUtils.expand_from_8x8(
                compressed_wm=recovered_watermark.astype(np.uint8),  # Ensure integer type
                original_shape=size,
                k=int(np.log2(number_of_sequences))
            )
        # Save recovered watermark
        recovered_watermark_path = os.path.join(
            self.directories["extracted_watermark_path"],
            f"Recovered_Watermark_{image_name}_gain_{gain_factor}_type{watermark_type}_{number_of_sequences}.bmp"
        )
        ImageUtils.save_image(img=recovered_watermark, path=recovered_watermark_path)

        # Calculate detection accuracy
        detection_accuracy = np.mean(recovered_watermark == original_watermark)

        print(f"{BColors.OK_GREEN}Watermark Extraction completed.{BColors.ENDC}")
        print(f"{BColors.OK_GREEN}Detection Accuracy: {detection_accuracy * 100:.2f}%{BColors.ENDC}")

        # Visualize watermark comparison
        comparison_path = os.path.join(
            self.directories["watermark_visualization_path"],
            f"Watermark_Comparison_{image_name}_gain_{gain_factor}_type{watermark_type}_{number_of_sequences}.png"
        )
        Visualization.visualize_watermark_comparison(
            original=original_watermark,
            recovered=recovered_watermark,
            image_name=image_name,
            gain_factor=gain_factor,
            watermark_type=watermark_type,
            save_path=comparison_path
        )
        return recovered_watermark, detection_accuracy
