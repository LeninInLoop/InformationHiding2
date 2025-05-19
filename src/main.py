import os

from watermarkAlgorithm import TwoLevelDCTWatermarkExtraction, TwoLevelDCTWatermarkEmbedding
from analyzer import AttackAnalyzer
from helpers import OptimalSeedFinder

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
        "results": "Images/Results",
    }

    # Initialize Two-Level DCT Watermarking
    embedder = TwoLevelDCTWatermarkEmbedding(directories)
    extractor = TwoLevelDCTWatermarkExtraction(directories)

    # Process image with watermark           G = 30                            G = 15
    watermark_type = 1                  # Watermark Type 2                # Watermark Type 2
    seed = 2825                         # GoldHill = 104, Lenna = 2537    # GoldHill = 800(95.31%), Lenna = 1419(93.75%)
    gain_factor = 30                    # Watermark Type 1                # Watermark Type 1
    image_name = "lenna"                # GoldHill = 292, Lenna = 2825    # GoldHill = 4891(96.88%), Lenna = 765(92.19%)
    equal_probability = False
    n_sequences = 4
    watermarked_image = embedder.embed_watermark(
        image_name=image_name,
        watermark_type=watermark_type,
        gain_factor=gain_factor,
        seed=seed,
        equal_probability=equal_probability,
        number_of_sequences=n_sequences
    )

    # Extract and verify watermark
    recovered_watermark, accuracy = extractor.extract_watermark(
        watermarked_image=watermarked_image,
        image_name=image_name,
        gain_factor=gain_factor,
        seed=seed,
        watermark_type=watermark_type,
        number_of_sequences=n_sequences
    )

    # watermark = WatermarkUtils.create_watermark_image(save_path = "Images/test.png", watermark_type = 2, size=(16, 16))
    # print(watermark)
    # compressed_watermark = WatermarkUtils.compress_to_8x8(wm=watermark, k=4)
    # uncompressed_watermark = WatermarkUtils.expand_from_8x8(
    #     compressed_wm=compressed_watermark,
    #     original_shape=(16, 16),
    #     k=4
    # )
    # print(50 * "=")
    # print(uncompressed_watermark)
    # accuracy = np.mean(watermark == uncompressed_watermark)

    print(f"Watermark detection accuracy: {accuracy * 100:.2f}%")

    watermarker = {
        "embedder": embedder,
        "extractor": extractor
    }
    AttackAnalyzer.analyze_gain_factor_psnr_relationship(
        watermarker=watermarker,
        image_names=["lenna", "goldhill"],
        seed=seed,
        n_sequences=n_sequences,
        watermark_type=watermark_type,
        equal_probability=equal_probability,
        save_path=os.path.join(
            directories["results"], f"{image_name}_{seed}_{watermark_type}_"
            f"gain_factor_psnr_relationship_{equal_probability}_{n_sequences}.png"
        )
    )
    AttackAnalyzer.analyze_gain_factor_correlation_relationship(
        watermarker=watermarker,
        image_names=["lenna", "goldhill"],
        seed=seed,
        n_sequences=n_sequences,
        equal_probability=equal_probability,
        watermark_type=watermark_type,
        save_path=os.path.join(
            directories["results"], f"{image_name}_{seed}_{watermark_type}_"
            f"gain_factor_NC_relationship_{equal_probability}_{n_sequences}.png"
        )
    )
    for image_name in ["lenna", "goldhill"]:
        AttackAnalyzer.analyze_jpeg_quality_correlation(
            watermarker=watermarker,
            image_name=image_name,
            seed=seed,
            n_sequences=n_sequences,
            watermark_type=watermark_type,
            equal_probability=equal_probability,
            save_path=os.path.join(
                directories["results"], f"{image_name}_{seed}_{watermark_type}_"
                f"gain_factor_JPEG_relationship_{equal_probability}_{n_sequences}.png"
            )
        )
        AttackAnalyzer.analyze_gaussian_noise_correlation(
            watermarker=watermarker,
            image_name=image_name,
            seed=seed,
            equal_probability=equal_probability,
            n_sequences=n_sequences,
            watermark_type=watermark_type,
            save_path=os.path.join(
                directories["results"], f"{image_name}_{seed}_{watermark_type}_"
                f"gain_factor_noise_relationship_{equal_probability}_{n_sequences}.png"
            )
        )
    #
    # OptimalSeedFinder.batch_test_seeds(
    #     watermarker=watermarker,
    #     image_names=["goldhill"],
    #     watermark_types=[2],
    #     gain_factors=[15],
    #     seed_ranges=(1, 100000, 5000)
    # )

if __name__ == '__main__':
    main()