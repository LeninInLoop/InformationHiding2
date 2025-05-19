from .generalHelper import BColors

class OptimalSeedFinder:
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
        """Find optimal seed for watermarking with early stopping on perfect match"""
        print(f"{BColors.HEADER}Finding optimal seed for image: {image_name}{BColors.ENDC}")
        print(f"{BColors.OK_BLUE}Parameters: watermark_type={watermark_type}, gain_factor={gain_factor}{BColors.ENDC}")
        print(f"{BColors.OK_BLUE}Testing seeds from {start_seed} to {max_seed}{BColors.ENDC}")

        best_seed = -1
        best_accuracy = 0.0
        attempts = 0

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
                    watermark_type=watermark_type,
                )

                print(f"{BColors.OK_BLUE}Seed {seed} achieved accuracy: {accuracy * 100:.2f}%{BColors.ENDC}")

                # Update the best seed if current is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_seed = seed
                    print(
                        f"{BColors.OK_GREEN}New best seed found: {best_seed} with accuracy {best_accuracy * 100:.2f}%{BColors.ENDC}")

                # If we reached 100% accuracy, break
                if accuracy >= 0.999:  # Using 0.999 instead of 1.0 to account for floating point precision
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
        """Test multiple configurations to find optimal seeds"""
        start_seed, max_seed, max_attempts = seed_ranges
        results = {}

        for image_name in image_names:
            for watermark_type in watermark_types:
                for gain_factor in gain_factors:
                    config_key = f"{image_name}_type{watermark_type}_gain{gain_factor}"
                    print(
                        f"\n{BColors.HEADER}Testing configuration: {config_key}{BColors.ENDC}")

                    optimal_seed = OptimalSeedFinder.find_optimal_seed(
                        watermarker=watermarker,
                        image_name=image_name,
                        watermark_type=watermark_type,
                        gain_factor=gain_factor,
                        start_seed=start_seed,
                        max_seed=max_seed,
                        max_attempts=max_attempts
                    )

                    results[config_key] = optimal_seed

        # Print summary of results
        print(f"\n{BColors.HEADER}Summary of Optimal Seeds:{BColors.ENDC}")
        for config, seed in results.items():
            accuracy_text = "100%" if seed != -1 else "Failed"
            print(f"{BColors.OK_BLUE}{config}: Seed={seed}, Accuracy={accuracy_text}{BColors.ENDC}")

        return results