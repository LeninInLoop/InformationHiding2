from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import os
import numpy as np
import io
from functools import lru_cache
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils import ImageUtils
from src.helpers import BColors, Visualization


class AttackAnalyzer:
    @staticmethod
    def _ensure_directory_exists(directory_path):
        """Ensure the specified directory exists, create it if it doesn't."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

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
    def _load_original_watermark(watermarker: Dict, watermark_type: int, number_of_sequences: int) -> Tuple[
        np.ndarray, np.ndarray]:
        watermark_path = os.path.join(
            watermarker["embedder"].directories["watermark_path"],
            f"watermark_type{watermark_type}_{number_of_sequences}.bmp"
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
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
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
        recovered_watermark, accuracy = watermarker["extractor"].extract_watermark(
            watermarked_image=watermarked_image,
            image_name=image_name,
            gain_factor=gain_factor,
            seed=seed,
            watermark_type=watermark_type,
            equal_probability=equal_probability,
            number_of_sequences=n_sequences
        )

        return watermarked_image, psnr, accuracy, recovered_watermark

    @staticmethod
    def _calculate_correlation(
            original_watermark_float: np.ndarray,
            recovered_watermark: np.ndarray
    ) -> float:
        """Calculate normalized correlation between original and recovered watermarks"""
        recovered_watermark_float = recovered_watermark.astype(np.float64)
        return np.corrcoef(original_watermark_float.flatten(), recovered_watermark_float.flatten())[0, 1]

    @staticmethod
    def visualize_watermark_comparison(
            original: np.ndarray,
            recovered: np.ndarray,
            image_name: str,
            gain_factor: float,
            watermark_type: int = 1,
            save_path: str = "watermark_comparison.tif",
    ) -> None:
        """
        Visualize the comparison between original and recovered watermarks.

        Args:
            original: Original watermark as a numpy array
            recovered: Recovered watermark as a numpy array
            image_name: Name of the image
            gain_factor: Gain factor used for watermarking
            watermark_type: Type of watermark used
            save_path: Path to save the visualization
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original watermark
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Watermark')
        axes[0].axis('off')

        # Plot recovered watermark
        axes[1].imshow(recovered, cmap='gray')
        axes[1].set_title('Recovered Watermark')
        axes[1].axis('off')

        # Plot difference (XOR for binary images)
        if original.dtype == np.bool_ and recovered.dtype == np.bool_:
            diff = np.logical_xor(original, recovered)
        else:
            diff = np.abs(original.astype(float) - recovered.astype(float))

        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Difference (Error)')
        axes[2].axis('off')

        # Add overall title
        correlation = AttackAnalyzer._calculate_correlation(
            original.astype(np.float64), recovered
        )
        fig.suptitle(f'Watermark Comparison - {image_name}\n'
                     f'(Type: {watermark_type}, Gain: {gain_factor}, Correlation: {correlation:.4f})')

        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

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

        # Create output directory for watermark visualizations
        output_dir = "watermark_visualizations"
        AttackAnalyzer._ensure_directory_exists(output_dir)

        # Create PSNR log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        psnr_log_file = f"psnr_gain_factor_analysis_{timestamp}_ns{n_sequences}.txt"

        with open(psnr_log_file, "w") as log_file:
            log_file.write("# PSNR Analysis - Gain Factor Relationship\n")
            log_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(
                f"# Parameters: Watermark Type={watermark_type}, Seed={seed}, Equal Probability={equal_probability}, Sequences={n_sequences}\n\n")

            # Load original watermark for visualization
            original_watermark, _ = AttackAnalyzer._load_original_watermark(
                watermarker, watermark_type, n_sequences
            )

            # Process each image
            for image_name in image_names:
                print(f"{BColors.HEADER}Processing image: {image_name}{BColors.ENDC}")
                log_file.write(f"## Image: {image_name}\n")
                log_file.write("Gain Factor | PSNR (dB) | Accuracy (%)\n")
                log_file.write("-" * 35 + "\n")

                psnr_values = []

                # Test different gain factors
                for gain in gain_factors:
                    print(f"{BColors.OK_BLUE}Testing gain factor: {gain}{BColors.ENDC}")

                    # Embed watermark and calculate metrics
                    _, psnr, accuracy, recovered_watermark = AttackAnalyzer._embed_and_calculate_psnr(
                        watermarker, image_name, gain, watermark_type, seed, equal_probability, n_sequences
                    )

                    # Store result
                    psnr_values.append(psnr)
                    log_file.write(f"{gain:^11} | {psnr:^8.2f} | {accuracy * 100:^11.2f}\n")

                    print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")
                    print(f"{BColors.OK_GREEN}Watermark detection accuracy: {accuracy * 100:.2f}%{BColors.ENDC}")

                    # Visualize watermark comparison
                    vis_save_path = os.path.join(output_dir, f"watermark_{image_name}_gain{gain}_{watermark_type}_ns{n_sequences}.png")
                    AttackAnalyzer.visualize_watermark_comparison(
                        original_watermark,
                        recovered_watermark,
                        image_name,
                        gain,
                        watermark_type,
                        vis_save_path
                    )

                results[image_name] = psnr_values
                log_file.write("\n\n")

        print(f"PSNR values saved to {psnr_log_file}")

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
            seed: Tuple = 48,
            save_path: str = "gain_factor_correlation_relationship.png",
            equal_probability: bool = False,
            n_sequences: int = 2
    ) -> Image.Image:
        # Use default images if none provided
        image_names = image_names or ["lenna", "goldhill"]
        results = {}

        # Create output directory for watermark visualizations
        output_dir = "watermark_visualizations"
        AttackAnalyzer._ensure_directory_exists(output_dir)

        # Create correlation log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        correlation_log_file = f"correlation_gain_factor_analysis_{timestamp}_ns{n_sequences}.txt"

        # Load original watermark
        original_watermark, orig_watermark_float = AttackAnalyzer._load_original_watermark(
            watermarker,
            watermark_type,
            number_of_sequences=n_sequences
        )

        with open(correlation_log_file, "w") as log_file:
            log_file.write("# Correlation Analysis - Gain Factor Relationship\n")
            log_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(
                f"# Parameters: Watermark Type={watermark_type}, Equal Probability={equal_probability}, Sequences={n_sequences}\n\n")

            # Process each image
            for image_name in image_names:
                current_seed = seed[0] if image_name == "lenna" else seed[1]
                print(f"{BColors.HEADER}Processing image: {image_name} (Seed: {current_seed}){BColors.ENDC}")
                log_file.write(f"## Image: {image_name} (Seed: {current_seed})\n")
                log_file.write("Gain Factor | Correlation | Accuracy (%)\n")
                log_file.write("-" * 40 + "\n")

                correlation_values = []

                # Test different gain factors
                for gain in gain_factors:
                    print(f"{BColors.OK_BLUE}Testing gain factor: {gain}{BColors.ENDC}")

                    # Embed watermark
                    watermarked_image = watermarker["embedder"].embed_watermark(
                        image_name=image_name,
                        watermark_type=watermark_type,
                        gain_factor=gain,
                        seed=seed[0] if image_name == "lenna" else seed[1],
                        equal_probability=equal_probability,
                        number_of_sequences=n_sequences
                    )

                    # Extract watermark
                    recovered_watermark, accuracy = watermarker["extractor"].extract_watermark(
                        watermarked_image=watermarked_image,
                        image_name=image_name,
                        gain_factor=gain,
                        seed=seed[0] if image_name == "lenna" else seed[1],
                        watermark_type=watermark_type,
                        equal_probability=equal_probability,
                        number_of_sequences=n_sequences
                    )

                    # Calculate normalized correlation
                    correlation = AttackAnalyzer._calculate_correlation(orig_watermark_float,
                                                                                recovered_watermark)

                    # Store result
                    correlation_values.append(correlation)
                    log_file.write(f"{gain:^11} | {correlation:^11.4f} | {accuracy * 100:^11.2f}\n")

                    print(f"{BColors.OK_GREEN}Normalized Correlation: {correlation:.4f}{BColors.ENDC}")
                    print(f"{BColors.OK_GREEN}Watermark detection accuracy: {accuracy * 100:.2f}%{BColors.ENDC}")

                    # Visualize watermark comparison
                    vis_save_path = os.path.join(output_dir,
                                                 f"watermark_corr_{image_name}_gain{gain}_{watermark_type}_ns{n_sequences}.png")
                    AttackAnalyzer.visualize_watermark_comparison(
                        original_watermark,
                        recovered_watermark,
                        image_name,
                        gain,
                        watermark_type,
                        vis_save_path
                    )

                results[image_name] = correlation_values
                log_file.write("\n\n")

        print(f"Correlation values saved to {correlation_log_file}")

        # Create plot
        title = (f'Relationship between Gain Factor and Correlation\n'
                 f'(Watermark Type: {watermark_type}, '
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
            quality_factors: range = range(1, 101, 11),
            watermark_type: int = 2,
            seed: int = 48,
            save_path: str = "jpeg_quality_correlation_relationship.png",
            equal_probability: bool = False,
            n_sequences: int = 2
    ) -> Image.Image:
        # Dictionary to store results
        gain_factors = gain_factors or [5, 10, 15, 20, 25, 30, 35]
        results = {}

        # Create output directory for watermark visualizations
        output_dir = "watermark_visualizations/jpeg_quality"
        AttackAnalyzer._ensure_directory_exists(output_dir)

        # Create log file for JPEG analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jpeg_log_file = f"jpeg_quality_analysis_{timestamp}_ns{n_sequences}.txt"

        # Load original watermark
        original_watermark, orig_watermark_float = AttackAnalyzer._load_original_watermark(
            watermarker,
            watermark_type,
            number_of_sequences=n_sequences
        )

        with open(jpeg_log_file, "w") as log_file:
            log_file.write("# JPEG Quality Compression Analysis\n")
            log_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"# Parameters: Image={image_name}, Watermark Type={watermark_type}, Seed={seed}, ")
            log_file.write(f"Equal Probability={equal_probability}, Sequences={n_sequences}\n\n")

            # Process each gain factor
            for gain in gain_factors:
                print(f"{BColors.HEADER}Processing gain factor: {gain}{BColors.ENDC}")
                log_file.write(f"## Gain Factor: {gain}\n")
                log_file.write("Quality | Correlation | PSNR (dB)\n")
                log_file.write("-" * 35 + "\n")

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

                # Get original image for PSNR calculation
                original_image = AttackAnalyzer.prepare_image_for_analysis(
                    watermarker["embedder"], image_name
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

                    ImageUtils.save_image(
                        os.path.join(output_dir, f"compressed_{image_name}_{quality}_{gain}.jpg"),
                        compressed_image
                    )
                    # Calculate PSNR between original and compressed
                    mse = np.mean((original_image - compressed_image) ** 2)
                    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

                    # Extract watermark from compressed image
                    recovered_watermark, accuracy = watermarker["extractor"].extract_watermark(
                        watermarked_image=compressed_image,
                        image_name=image_name,
                        gain_factor=gain,
                        seed=seed,
                        watermark_type=watermark_type,
                        equal_probability=equal_probability,
                        number_of_sequences=n_sequences
                    )

                    # Calculate normalized correlation
                    correlation = AttackAnalyzer._calculate_correlation(orig_watermark_float,
                                                                                recovered_watermark)

                    # Store result
                    correlation_values.append(correlation)
                    log_file.write(f"{quality:^7} | {correlation:^11.4f} | {psnr:^8.2f}\n")

                    print(f"{BColors.OK_GREEN}Normalized Correlation: {correlation:.4f}{BColors.ENDC}")
                    print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")

                    # Visualize watermark for selected qualities (to avoid too many images)
                    vis_save_path = os.path.join(output_dir,
                                                 f"watermark_jpeg_{image_name}_gain{gain}_q{quality}_ns{n_sequences}.png")
                    AttackAnalyzer.visualize_watermark_comparison(
                        original_watermark,
                        recovered_watermark,
                        f"{image_name} (JPEG Q{quality})",
                        gain,
                        watermark_type,
                        vis_save_path
                    )

                results[f'G: {gain}'] = correlation_values
                log_file.write("\n\n")

        print(f"JPEG quality analysis saved to {jpeg_log_file}")

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

        # Create output directory for watermark visualizations
        output_dir = "watermark_visualizations/gaussian_noise"
        AttackAnalyzer._ensure_directory_exists(output_dir)

        # Create log file for noise analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_log_file = f"gaussian_noise_analysis_{timestamp}.txt"

        # Load original watermark
        original_watermark, orig_watermark_float = AttackAnalyzer._load_original_watermark(
            watermarker,
            watermark_type,
            number_of_sequences=n_sequences
        )

        with open(noise_log_file, "w") as log_file:
            log_file.write("# Gaussian Noise Analysis\n")
            log_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"# Parameters: Image={image_name}, Watermark Type={watermark_type}, Seed={seed}, ")
            log_file.write(f"Equal Probability={equal_probability}, Sequences={n_sequences}\n\n")

            # Get original image for PSNR calculation
            original_image = AttackAnalyzer.prepare_image_for_analysis(
                watermarker["embedder"], image_name
            )

            # Process each gain factor
            for gain in gain_factors:
                print(f"{BColors.HEADER}Processing gain factor: {gain}{BColors.ENDC}")
                log_file.write(f"## Gain Factor: {gain}\n")
                log_file.write("Noise % | Correlation | PSNR (dB)\n")
                log_file.write("-" * 35 + "\n")

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
                    print(f"{BColors.OK_BLUE}Testing noise level: {noise_level}%{BColors.ENDC}")

                    # Apply noise (or not if noise_level = 0)
                    if noise_level == 0:
                        noisy_image = watermarked_image.copy()
                        psnr = float('inf')  # No noise, so infinite PSNR
                    else:
                        # Add Gaussian noise as a percentage of the image's dynamic range
                        rng = np.random.RandomState(seed)  # For reproducibility

                        # Calculate noise scale based on percentage
                        # Assuming image is in range [0, 255]
                        noise_scale = (noise_level / 100.0) * 255.0

                        # Generate and apply noise
                        noise = rng.normal(0, noise_scale, watermarked_image.shape)
                        noisy_image = np.clip(watermarked_image + noise, 0, 255).astype(watermarked_image.dtype)

                        # Calculate PSNR between original and noisy image
                        mse = np.mean((original_image - noisy_image) ** 2)
                        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

                        # Save the noisy image
                        noisy_path = os.path.join(
                            watermarker["embedder"].directories["noisy_images"],
                            f"{image_name}_Noisy_image_{noise_level}_ns{n_sequences}.png"
                        )
                        ImageUtils.save_image(noisy_path, noisy_image)

                    # Extract watermark from noisy image
                    recovered_watermark, accuracy = watermarker["extractor"].extract_watermark(
                        watermarked_image=noisy_image,
                        image_name=image_name,
                        gain_factor=gain,
                        seed=seed,
                        watermark_type=watermark_type,
                        equal_probability=equal_probability,
                        number_of_sequences=n_sequences
                    )

                    # Calculate normalized correlation
                    correlation = AttackAnalyzer._calculate_correlation(orig_watermark_float,
                                                                                recovered_watermark)

                    # Store result
                    correlation_values.append(correlation)
                    log_file.write(f"{noise_level:^7} | {correlation:^11.4f} | {psnr:^8.2f}\n")

                    print(f"{BColors.OK_GREEN}Normalized Correlation: {correlation:.4f}{BColors.ENDC}")
                    print(f"{BColors.OK_GREEN}PSNR: {psnr:.2f} dB{BColors.ENDC}")

                    # Visualize watermark for selected noise levels
                    if noise_level in [0, 15, 30, 50] or noise_level == noise_levels[-1]:
                        vis_save_path = os.path.join(output_dir,
                                                     f"watermark_noise_{image_name}_gain{gain}_n{noise_level}_ns{n_sequences}.png")
                        AttackAnalyzer.visualize_watermark_comparison(
                            original_watermark,
                            recovered_watermark,
                            f"{image_name} (Noise {noise_level}%)",
                            gain,
                            watermark_type,
                            vis_save_path
                        )

                results[f'Gain Factor: {gain}'] = correlation_values
                log_file.write("\n\n")

        print(f"Gaussian noise analysis saved to {noise_log_file}")

        # Create plot
        title = (f'Relationship between Gaussian noise and correlation\n'
                 f'for {image_name} (Master Seed: {seed}, Watermark Type: {watermark_type}, '
                 f'Equal Probability: {equal_probability})')

        return Visualization.plot_relationship(
            noise_levels, results, title, '% Gaussian Noise', 'Correlation',
            save_path, ylim=(0, 1.0), legend_loc='upper right'
        )