from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import os
import numpy as np
import io
from functools import lru_cache

from src.utils import ImageUtils
from src.helpers import BColors, Visualization


class AttackAnalyzer:
    @staticmethod
    @lru_cache(maxsize=32)
    def prepare_image_for_analysis(watermarker, image_name: str) -> np.ndarray:
        original_path = os.path.join(watermarker.directories['image_base_path'], f"{image_name}.bmp")
        original_image = ImageUtils.load_image(original_path)

        # Convert to grayscale if needed
        if len(original_image.shape) > 2:
            original_image = ImageUtils.convert_to_gray_scale(original_image)

        # Resize to standard size
        return ImageUtils.resize_image(original_image, (512, 512))

    @staticmethod
    def _load_original_watermark(watermarker: Dict, watermark_type: int) -> Tuple[np.ndarray, np.ndarray]:
        watermark_path = os.path.join(
            watermarker["embedder"].directories["watermark_path"],
            f"watermark_type{watermark_type}.bmp"
        )
        original_watermark = ImageUtils.load_image(watermark_path).astype(np.bool_)
        return original_watermark, original_watermark.astype(np.float64)

    @staticmethod
    def _embed_and_calculate_psnr(
            watermarker: Dict,
            image_name: str,
            gain_factor: int,
            watermark_type: int,
            seed: int,
            equal_probability: bool,
            n_sequences: int
    ) -> Tuple[np.ndarray, float, float]:
        # Get original image
        original_image = AttackAnalyzer.prepare_image_for_analysis(watermarker["embedder"], image_name)

        # Embed watermark
        watermarked_image = watermarker["embedder"].embed_watermark(
            image_name=image_name,
            watermark_type=watermark_type,
            gain_factor=gain_factor,
            seed=seed,
            equal_probability=equal_probability,
            number_of_sequences=n_sequences
        )

        # Calculate PSNR
        mse = np.mean((original_image - watermarked_image) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

        # Extract watermark to check accuracy
        _, accuracy = watermarker["extractor"].extract_watermark(
            watermarked_image=watermarked_image,
            image_name=image_name,
            gain_factor=gain_factor,
            seed=seed,
            watermark_type=watermark_type,
            equal_probability=equal_probability,
            number_of_sequences=n_sequences
        )

        return watermarked_image, psnr, accuracy

    @staticmethod
    def _calculate_correlation(
            original_watermark_float: np.ndarray,
            recovered_watermark: np.ndarray
    ) -> float:
        """Calculate normalized correlation between original and recovered watermarks"""
        recovered_watermark_float = recovered_watermark.astype(np.float64)
        return np.corrcoef(original_watermark_float.flatten(), recovered_watermark_float.flatten())[0, 1]

    @staticmethod
    def analyze_gain_factor_psnr_relationship(
            watermarker: Dict,
            image_names: Optional[List[str]] = None,
            gain_factors: range = range(5, 36, 5),
            watermark_type: int = 2,
            seed: int = 48,
            save_path: str = "gain_factor_psnr_relationship.png",
            equal_probability: bool = False,
            n_sequences: int = 2
    ) -> Image.Image:
        # Use default images if none provided
        image_names = image_names or ["goldhill", "lenna"]
        results = {}

        # Process each image
        for image_name in image_names:
            print(f"{BColors.HEADER}Processing image: {image_name}{BColors.ENDC}")
            psnr_values = []

            # Test different gain factors
            for gain in gain_factors:
                print(f"{BColors.OK_BLUE}Testing gain factor: {gain}{BColors.ENDC}")

                # Embed watermark and calculate metrics
                _, psnr, accuracy = AttackAnalyzer._embed_and_calculate_psnr(
                    watermarker, image_name, gain, watermark_type, seed, equal_probability, n_sequences
                )

                # Store result
                psnr_values.append(psnr)
                print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")
                print(f"{BColors.OK_GREEN}Watermark detection accuracy: {accuracy * 100:.2f}%{BColors.ENDC}")

            results[image_name] = psnr_values

        # Create plot
        title = (f'Relationship between Gain Factor and PSNR\n'
                 f'(Master Seed: {seed}, Watermark Type: {watermark_type}, '
                 f'Equal Probability: {equal_probability})')

        return Visualization.plot_relationship(
            gain_factors, results, title, 'Gain Factor', 'PSNR (dB)', save_path
        )

    @staticmethod
    def analyze_gain_factor_correlation_relationship(
            watermarker: Dict,
            image_names: Optional[List[str]] = None,
            gain_factors: range = range(5, 36, 5),
            watermark_type: int = 2,
            seed: int = 48,
            save_path: str = "gain_factor_correlation_relationship.png",
            equal_probability: bool = False,
            n_sequences: int = 2
    ) -> Image.Image:
        # Use default images if none provided
        image_names = image_names or ["lenna", "goldhill"]
        results = {}

        # Load original watermark
        _, orig_watermark_float = AttackAnalyzer._load_original_watermark(watermarker, watermark_type)

        # Process each image
        for image_name in image_names:
            print(f"{BColors.HEADER}Processing image: {image_name}{BColors.ENDC}")
            correlation_values = []

            # Test different gain factors
            for gain in gain_factors:
                print(f"{BColors.OK_BLUE}Testing gain factor: {gain}{BColors.ENDC}")

                # Embed watermark
                watermarked_image = watermarker["embedder"].embed_watermark(
                    image_name=image_name,
                    watermark_type=watermark_type,
                    gain_factor=gain,
                    seed=seed,
                    equal_probability=equal_probability,
                    number_of_sequences=n_sequences
                )

                # Extract watermark
                recovered_watermark, accuracy = watermarker["extractor"].extract_watermark(
                    watermarked_image=watermarked_image,
                    image_name=image_name,
                    gain_factor=gain,
                    seed=seed,
                    watermark_type=watermark_type,
                    equal_probability=equal_probability,
                    number_of_sequences=n_sequences
                )

                # Calculate normalized correlation
                correlation = AttackAnalyzer._calculate_correlation(orig_watermark_float, recovered_watermark)

                # Store result
                correlation_values.append(correlation)
                print(f"{BColors.OK_GREEN}Normalized Correlation: {correlation:.4f}{BColors.ENDC}")
                print(f"{BColors.OK_GREEN}Watermark detection accuracy: {accuracy * 100:.2f}%{BColors.ENDC}")

            results[image_name] = correlation_values

        # Create plot
        title = (f'Relationship between Gain Factor and Correlation\n'
                 f'(Master Seed: {seed}, Watermark Type: {watermark_type}, '
                 f'Equal Probability: {equal_probability})')

        return Visualization.plot_relationship(
            gain_factors, results, title, 'Gain Factor', 'Correlation Index',
            save_path, ylim=(0, 1.0), legend_loc='lower right'
        )

    @staticmethod
    def analyze_jpeg_quality_correlation(
            watermarker: Dict,
            image_name: str = "lenna",
            gain_factors: Optional[List[int]] = None,
            quality_factors: range = range(10, 101, 10),
            watermark_type: int = 2,
            seed: int = 48,
            save_path: str = "jpeg_quality_correlation_relationship.png",
            equal_probability: bool = False,
            n_sequences: int = 2
    ) -> Image.Image:
        # Dictionary to store results
        gain_factors = gain_factors or [5, 10, 15, 20, 25, 30, 35]
        results = {}

        # Load original watermark
        _, orig_watermark_float = AttackAnalyzer._load_original_watermark(watermarker, watermark_type)

        # Process each gain factor
        for gain in gain_factors:
            print(f"{BColors.HEADER}Processing gain factor: {gain}{BColors.ENDC}")
            correlation_values = []

            # Embed watermark
            watermarked_image = watermarker["embedder"].embed_watermark(
                image_name=image_name,
                watermark_type=watermark_type,
                gain_factor=gain,
                seed=seed,
                equal_probability=equal_probability,
                number_of_sequences=n_sequences
            )

            # Test different JPEG quality factors
            for quality in quality_factors:
                print(f"{BColors.OK_BLUE}Testing JPEG quality: {quality}{BColors.ENDC}")

                # Apply JPEG compression
                img_pil = Image.fromarray(watermarked_image.astype(np.uint8))
                jpeg_buffer = io.BytesIO()
                img_pil.save(jpeg_buffer, format="JPEG", quality=quality)
                jpeg_buffer.seek(0)
                compressed_image = np.array(Image.open(jpeg_buffer), dtype=np.float32)

                # Extract watermark from compressed image
                recovered_watermark, _ = watermarker["extractor"].extract_watermark(
                    watermarked_image=compressed_image,
                    image_name=image_name,
                    gain_factor=gain,
                    seed=seed,
                    watermark_type=watermark_type,
                    equal_probability=equal_probability,
                    number_of_sequences=n_sequences
                )

                # Calculate normalized correlation
                correlation = AttackAnalyzer._calculate_correlation(orig_watermark_float, recovered_watermark)

                # Store result
                correlation_values.append(correlation)
                print(f"{BColors.OK_GREEN}Normalized Correlation: {correlation:.4f}{BColors.ENDC}")

            results[f'G: {gain}'] = correlation_values

        # Create plot
        title = (f'Relationship between JPEG quality factor and correlation\n'
                 f'for {image_name} (Master Seed: {seed}, Watermark Type: {watermark_type}, '
                 f'Equal Probability: {equal_probability})')

        return Visualization.plot_relationship(
            quality_factors, results, title, 'JPEG Quality Factor', 'Correlation Index',
            save_path, ylim=(0, 1.0), legend_loc='lower right'
        )

    @staticmethod
    def analyze_gaussian_noise_correlation(
            watermarker: Dict,
            image_name: str = "lenna",
            gain_factors: Optional[List[int]] = None,
            noise_levels: range = range(0, 51, 5),
            watermark_type: int = 2,
            seed: int = 48,
            save_path: str = "gaussian_noise_correlation_relationship.png",
            equal_probability: bool = False,
            n_sequences: int = 2
    ) -> Image.Image:
        # Dictionary to store results
        gain_factors = gain_factors or [10, 15, 20, 25, 30, 35]
        results = {}

        # Load original watermark
        _, orig_watermark_float = AttackAnalyzer._load_original_watermark(watermarker, watermark_type)

        # Process each gain factor
        for gain in gain_factors:
            print(f"{BColors.HEADER}Processing gain factor: {gain}{BColors.ENDC}")
            correlation_values = []

            # Embed watermark
            watermarked_image = watermarker["embedder"].embed_watermark(
                image_name=image_name,
                watermark_type=watermark_type,
                gain_factor=gain,
                seed=seed,
                equal_probability=equal_probability,
                number_of_sequences=n_sequences
            )

            # Test different noise levels
            for noise_level in noise_levels:
                print(f"{BColors.OK_BLUE}Testing noise level: {noise_level}{BColors.ENDC}")

                # Apply noise (or not if noise_level = 0)
                if noise_level == 0:
                    noisy_image = watermarked_image.copy()
                else:
                    # Add Gaussian noise
                    rng = np.random.RandomState(seed)  # For reproducibility
                    noise = rng.normal(0, noise_level, watermarked_image.shape)
                    noisy_image = np.clip(watermarked_image + noise, 0, 255)

                # Extract watermark from noisy image
                recovered_watermark, _ = watermarker["extractor"].extract_watermark(
                    watermarked_image=noisy_image,
                    image_name=image_name,
                    gain_factor=gain,
                    seed=seed,
                    watermark_type=watermark_type,
                    equal_probability=equal_probability,
                    number_of_sequences=n_sequences
                )

                # Calculate normalized correlation
                correlation = AttackAnalyzer._calculate_correlation(orig_watermark_float, recovered_watermark)

                # Store result
                correlation_values.append(correlation)
                print(f"{BColors.OK_GREEN}Normalized Correlation: {correlation:.4f}{BColors.ENDC}")

            results[f'Gain Factor: {gain}'] = correlation_values

        # Create plot
        title = (f'Relationship between Gaussian noise and correlation\n'
                 f'for {image_name} (Master Seed: {seed}, Watermark Type: {watermark_type}, '
                 f'Equal Probability: {equal_probability})')

        return Visualization.plot_relationship(
            noise_levels, results, title, 'σ Gaussian Noise', 'Correlation Index',
            save_path, ylim=(0, 1.0), legend_loc='upper right'
        )