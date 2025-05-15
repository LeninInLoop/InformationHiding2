import numpy as np
import random

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


class LowCorrelationSequenceGenerator:
    """Class for generating sequences with low correlation."""

    def __init__(self, master_seed=42):
        self.master_rng = random.Random(master_seed)
        self.best_seeds = (None, None)
        self.best_sequences = (None, None)
        self.best_correlation = 1.0

    def set_master_seed(self, master_seed):
        self.master_rng = random.Random(master_seed)

    @staticmethod
    def generate_sequence(seed, length=39):
        rng = random.Random(seed)
        return [rng.choice([-1, 0, 1]) for _ in range(length)]

    @classmethod
    def verify_sequences(cls, seed1, seed2, length=39):
        seq1 = cls.generate_sequence(seed1, length)
        seq2 = cls.generate_sequence(seed2, length)
        correlation = np.corrcoef(seq1, seq2)[0, 1]
        return seq1, seq2, correlation

    def find_low_correlation_seeds(
            self,
            target_correlation=0.05,
            max_attempts=10000,
            sequence_length=39,
            min_seed=1,
            max_seed=100000,
            verbose=True
    ):
        self.best_correlation = 1.0
        self.best_seeds = (None, None)
        self.best_sequences = (None, None)

        # Try random seed combinations
        for attempt in range(max_attempts):
            seed1 = self.master_rng.randint(min_seed, max_seed)
            seed2 = self.master_rng.randint(min_seed, max_seed)

            # Skip if we're trying the same seed twice
            if seed1 == seed2:
                continue

            # Generate sequences
            seq1 = self.generate_sequence(seed1, sequence_length)
            seq2 = self.generate_sequence(seed2, sequence_length)

            # Calculate correlation using NumPy for consistency
            correlation = np.corrcoef(seq1, seq2)[0, 1]
            correlation_abs = abs(correlation)

            # Update if this is better than our previous best
            if correlation_abs < self.best_correlation:
                self.best_correlation = correlation_abs
                self.best_seeds = (seed1, seed2)
                self.best_sequences = (seq1, seq2)

                # Print progress updates if verbose
                if verbose:
                    print(BColors.WARNING + f"Attempt {attempt}: Found seeds with correlation {correlation:.6f} "
                          f"(abs: {correlation_abs:.6f}): {self.best_seeds}" + BColors.ENDC)

                # Check if we've met our target
                if correlation_abs < target_correlation:
                    break

        # Verify the results once more
        final_seq1, final_seq2, final_correlation = self.verify_sequences(
            self.best_seeds[0], self.best_seeds[1], sequence_length
        )

        return self.best_seeds, (final_seq1, final_seq2), final_correlation

    def save_seed_results(self, filename):
        if self.best_seeds[0] is None or self.best_seeds[1] is None:
            raise ValueError("No seeds found yet. Run find_low_correlation_seeds first.")

        with open(filename, 'w') as f:
            f.write(f"Seeds: {self.best_seeds[0]}, {self.best_seeds[1]}\n")
            f.write(f"Correlation: {self.best_correlation:.6f}\n\n")
            f.write(f"Sequence 1: {self.best_sequences[0]}\n\n")
            f.write(f"Sequence 2: {self.best_sequences[1]}\n")

    def verify_reproducibility(self, sequence_length=39, verbose=True):
        if self.best_seeds[0] is None or self.best_seeds[1] is None:
            raise ValueError("No seeds found yet. Run find_low_correlation_seeds first.")

        seed1, seed2 = self.best_seeds
        reproduced_seq1, reproduced_seq2, reproduced_corr = self.verify_sequences(
            seed1, seed2, sequence_length
        )

        if verbose:
            print(f"Regenerated correlation: {reproduced_corr:.6f}")

        # Check if sequences are identical to original ones
        seq1_match = self.best_sequences[0] == reproduced_seq1
        seq2_match = self.best_sequences[1] == reproduced_seq2

        if verbose:
            print(f"Sequence 1 reproduced correctly: {seq1_match}")
            print(f"Sequence 2 reproduced correctly: {seq2_match}")

        return seq1_match and seq2_match, reproduced_corr

    def get_best_results(self):
        if self.best_seeds[0] is None or self.best_seeds[1] is None:
            raise ValueError("No seeds found yet. Run find_low_correlation_seeds first.")

        return self.best_seeds, self.best_sequences, self.best_correlation