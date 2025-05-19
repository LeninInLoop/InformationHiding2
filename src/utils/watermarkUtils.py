from typing import List, Tuple
import numpy as np

from src.helpers import BColors
from .seedFinder import MultiSequenceGenerator
from .imageUtils import ImageUtils

class WatermarkUtils:
    @staticmethod
    def compress_to_8x8(wm: np.ndarray, k: int) -> np.ndarray:
        """Compress binary array to 8x8 by grouping k bits"""
        flat = wm.flatten()
        required_size = 64 * k

        if flat.size != required_size:
            raise ValueError(
                f"For 8x8 compression with k={k}, original size must be "
                f"{required_size} elements (64×{k}), got {flat.size}"
            )
        compressed = np.zeros(64, dtype=np.uint8)
        for i in range(64):
            group = flat[i * k: (i + 1) * k]
            compressed[i] = sum(bit << (k - 1 - j) for j, bit in enumerate(group))
        return compressed.reshape((8, 8))

    @staticmethod
    def expand_from_8x8(compressed_wm: np.ndarray, original_shape: tuple, k: int) -> np.ndarray:
        """Expand a compressed array back to original binary form"""
        compressed_flat = compressed_wm.astype(np.uint8).flatten()
        target_size = np.prod(original_shape)
        if target_size != 64 * k:
            raise ValueError(
                f"Original shape {original_shape} ({target_size} elements) "
                f"must be 64×{k} = {64 * k} elements for k={k}"
            )
        bits = np.zeros(target_size, dtype=np.uint8)
        for i, val in enumerate(compressed_flat):
            for j in range(k):
                bits[i * k + j] = (val >> (k - 1 - j)) & 0x1
        return bits.reshape(original_shape)

    @staticmethod
    def create_watermark_image(save_path: str = None, watermark_type: int = 1, size: tuple = (8, 8)) -> np.ndarray:
        if size == (8, 8):
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
            else:  # watermark_type == 2 (or any other value)
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
        elif size == (8, 16):
            if watermark_type == 1:
                # Simple upscaled version for demonstration, or create a new pattern
                watermark_array = np.array([
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                ], dtype=np.bool_)
            else:  # watermark_type == 2
                watermark_array = np.array([
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                ], dtype=np.bool_)
        elif size == (12, 16):
            if watermark_type == 1:
                # Upscale the 8x8 pattern by 4
                watermark_array = np.array([
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                ], dtype=np.bool_)
            else:  # watermark_type == 2
                watermark_array = np.array([
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                ], dtype=np.bool_)
        elif size == (16, 16):
            if watermark_type == 1:
                # Upscale the 8x8 pattern by 4
                watermark_array = np.array([
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                ], dtype=np.bool_)
            else:  # watermark_type == 2
                watermark_array = np.array([
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                ], dtype=np.bool_)
        else:
            # Optionally handle unsupported sizes
            print(f"Warning: Watermark size {size} not explicitly supported. Returning None.")
            return None

        ImageUtils.save_image(img=watermark_array, path=save_path)
        return watermark_array

    @staticmethod
    def generate_noise(
            length,
            n_streams=2,
            seed=None,
            image_name: str = "lenna",
            watermark_type: int = 1,
            equal_probability: bool = False
    ) -> np.ndarray:
        generator = MultiSequenceGenerator(
            master_seed = seed
            )
        seeds, sequences, corr_matrix, max_abs_corr = generator.find_low_correlation_seeds(
            num_sequences=n_streams,
            target_correlation=0.003,
            max_attempts=10000,
            sequence_length=length,
            verbose=True,
            equal_probability=equal_probability
        )

        print(BColors.OK_GREEN + f"Best seeds found: {seeds}" + BColors.ENDC)
        print(BColors.OK_GREEN + f"Correlation: {corr_matrix}" + BColors.ENDC)

        is_reproducible, regenerated_corr = generator.verify_reproducibility()
        print(BColors.OK_GREEN + f"Sequences are reproducible: {is_reproducible}" + BColors.ENDC)

        generator.save_seed_results(f"{image_name}_{seed}_low_correlation_seeds_type{watermark_type}.txt")
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
        # Copy source blocks to avoid modifying original data
        result = dct_blocks.copy()
        # Retrieve indices of high-frequency coefficients to modify
        high_freq_indices = WatermarkUtils.get_high_frequency_indices()

        num_patterns = len(noise)
        # Iterate over each block
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                bits = watermark[i, j]
                if isinstance(bits, (bytes, bytearray)):
                    bits_str = bits.decode('ascii')
                    idx = int(bits_str, 2)
                elif isinstance(bits, str):
                    idx = int(bits, 2)
                elif isinstance(bits, (int, float)):
                    idx = int(bits)
                elif hasattr(bits, 'item'):
                    idx = int(bits.item())
                else:
                    raise TypeError(f"Unsupported watermark bit type: {type(bits)} at ({i},{j})")

                if num_patterns == 2:
                    idx = 1 if idx else 0
                else:
                    if not (0 <= idx < num_patterns):
                        raise ValueError(f"Watermark index {idx} out of range for noise patterns (0..{num_patterns-1}) at ({i},{j})")

                noise_pattern = noise[idx]

                # print(f"\nBlock ({i},{j}) – watermark bits/value = {bits} (pattern index = {idx})")

                # Embed by adjusting each high-frequency coefficient
                for k, (u, v) in enumerate(high_freq_indices):
                    before = result[i, j, u, v]
                    delta = gain_factor * noise_pattern[k]
                    result[i, j, u, v] = before + delta
                    after = result[i, j, u, v]
                    # print(f"  HF[{u},{v}]: {before:.4f} + {delta:.4f} → {after:.4f}")

        return result

    @staticmethod
    def calculate_correlation(block_dct_coeffs, noise_patterns, high_freq_indices):
        high_freq_values = np.array([block_dct_coeffs[u, v] for (u, v) in high_freq_indices])
        # print(f"\nExtracting from block – high-frequency coeffs:\n {high_freq_values}")

        correlations = np.zeros(len(noise_patterns), dtype=np.float64)
        for bit in range(len(noise_patterns)):
            noise_values = noise_patterns[bit]
            corr = np.corrcoef(high_freq_values, noise_values)[0,1]
            correlations[bit] = corr
            # print(f"  Corr with noise[{bit}]: {corr:.6f}")
        return correlations