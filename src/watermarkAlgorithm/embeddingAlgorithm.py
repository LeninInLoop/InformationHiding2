from typing import Dict
import numpy as np, os

from src.utils import ImageUtils, DCTUtils, WatermarkUtils
from src.helpers import BColors

class TwoLevelDCTWatermarkEmbedding:

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
            equal_probability: bool = False,
            number_of_sequences: int = 2
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
        watermark_path = os.path.join(self.directories["watermark_path"], f"watermark_type{watermark_type}.bmp")
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
        watermark = WatermarkUtils.create_watermark_image(
            save_path=watermark_path,
            watermark_type=watermark_type,
            size=size,
        )
        # print(watermark)
        if number_of_sequences != 2:
            watermark = WatermarkUtils.compress_to_8x8(
                wm = watermark,
                k = int(np.log2(number_of_sequences)),
            )
        print(watermark)
        print(f"{BColors.OK_BLUE}Watermark Size: {watermark.shape}{BColors.ENDC}")

        # Generate noise patterns for bit 0 and bit 1
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()
        noise = WatermarkUtils.generate_noise(
            length=len(high_freq_indices),
            n_streams=number_of_sequences,
            seed=seed,
            image_name=image_name,
            watermark_type=watermark_type,
            equal_probability=equal_probability
        )

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
            f"{image_name}_watermarked_lrai_gain_{gain_factor}_type{watermark_type}.bmp"
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
            f"Watermarked_{image_name}_gain_{gain_factor}_type{watermark_type}.bmp"
        )
        ImageUtils.save_image(img=watermarked_image, path=watermarked_path)

        # Calculate PSNR
        mse = np.mean((original_image - watermarked_image) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

        print(f"{BColors.OK_GREEN}Watermarking completed successfully!{BColors.ENDC}")
        print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")

        return watermarked_image
