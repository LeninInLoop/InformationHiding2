import numpy as np, random
from src.helpers import BColors

class MultiSequenceGenerator:
    """Class for generating multiple sequences with low mutual correlations."""

    def __init__(self, master_seed=42):
        self.master_rng = random.Random(master_seed)
        self.best_seeds = []
        self.best_sequences = []
        self.best_correlation_matrix = None
        self.best_max_abs_correlation = 1.0

    def set_master_seed(self, master_seed):
        self.master_rng = random.Random(master_seed)

    @staticmethod
    def generate_sequence(seed, length=39, equal_probability=False, symbols=None, weights=None):
        rng = random.Random(seed)

        # Default symbols if none provided
        if symbols is None:
            symbols = [-1, 0, 1]

        # If equal probability is requested, create equal weights
        if equal_probability:
            weights = [1 / len(symbols)] * len(symbols)

        # If weights are provided but equal_probability is False, use the provided weights
        if weights is not None and not equal_probability:
            # Normalize weights to sum to 1
            total = sum(weights)
            normalized_weights = [w / total for w in weights]
            return [rng.choices(symbols, weights=normalized_weights)[0] for _ in range(length)]

        # Use equal probability if specified, otherwise use default uniform choice
        if equal_probability:
            return [rng.choices(symbols, weights=weights)[0] for _ in range(length)]
        else:
            return [rng.choice(symbols) for _ in range(length)]

    @staticmethod
    def calculate_correlation_matrix(sequences):
        """Calculate the correlation matrix between all sequences."""
        num_sequences = len(sequences)
        correlation_matrix = np.zeros((num_sequences, num_sequences))

        for i in range(num_sequences):
            for j in range(i, num_sequences):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    correlation = np.corrcoef(sequences[i], sequences[j])[0, 1]
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation

        return correlation_matrix

    @staticmethod
    def get_max_abs_correlation(corr_matrix):
        """Get the maximum absolute correlation from the matrix, excluding self-correlations."""
        # Create a mask for the diagonal elements
        n = corr_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)

        # Get the maximum absolute value from the off-diagonal elements
        max_abs_corr = np.max(np.abs(corr_matrix[mask]))
        return max_abs_corr

    def find_low_correlation_seeds(
            self,
            num_sequences=2,
            target_correlation=0.05,
            max_attempts=10000,
            sequence_length=39,
            min_seed=1,
            max_seed=100000,
            verbose=True,
            equal_probability=False,
            symbols=None,
            weights=None
    ):
        """Find multiple seeds that generate sequences with low mutual correlations."""
        self.best_max_abs_correlation = 1.0
        self.best_seeds = []
        self.best_sequences = []
        self.best_correlation_matrix = None

        # Validate inputs
        if num_sequences < 2:
            raise ValueError("Number of sequences must be at least 2")

        # Try random seed combinations
        for attempt in range(max_attempts):
            # Generate random seeds
            seeds = []
            for _ in range(num_sequences):
                seed = self.master_rng.randint(min_seed, max_seed)
                while seed in seeds:  # Ensure unique seeds
                    seed = self.master_rng.randint(min_seed, max_seed)
                seeds.append(seed)

            # Generate sequences
            sequences = [
                self.generate_sequence(seed, sequence_length, equal_probability, symbols, weights)
                for seed in seeds
            ]

            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(sequences)

            # Get the maximum absolute correlation
            max_abs_corr = self.get_max_abs_correlation(corr_matrix)

            # Update if this is better than our previous best
            if max_abs_corr < self.best_max_abs_correlation:
                self.best_max_abs_correlation = max_abs_corr
                self.best_seeds = seeds
                self.best_sequences = sequences
                self.best_correlation_matrix = corr_matrix

                # Print progress updates if verbose
                if verbose:
                    print(BColors.WARNING + f"Attempt {attempt}: Found seeds with max correlation {max_abs_corr:.6f} "
                                            f"Seeds: {seeds}" + BColors.ENDC)

                # Check if we've met our target
                if max_abs_corr < target_correlation:
                    break

        if verbose:
            print(BColors.OK_GREEN + f"\nBest result found:")
            print(f"Maximum absolute correlation: {self.best_max_abs_correlation:.6f}")
            print(f"Seeds: {self.best_seeds}" + BColors.ENDC)

        return self.best_seeds, self.best_sequences, self.best_correlation_matrix, self.best_max_abs_correlation

    def verify_sequences(
            self,
            seeds,
            sequence_length=39,
            equal_probability=False,
            symbols=None,
            weights=None
    ):
        """Verify that the sequences generated from seeds have low correlations."""
        sequences = [
            self.generate_sequence(seed, sequence_length, equal_probability, symbols, weights)
            for seed in seeds
        ]
        corr_matrix = self.calculate_correlation_matrix(sequences)
        max_abs_corr = self.get_max_abs_correlation(corr_matrix)

        return sequences, corr_matrix, max_abs_corr

    def save_seed_results(self, filename):
        """Save the seeds, sequences, and correlation matrix to a file."""
        if not self.best_seeds:
            raise ValueError("No seeds found yet. Run find_low_correlation_seeds first.")

        with open(filename, 'w') as f:
            f.write(f"Seeds: {self.best_seeds}\n")
            f.write(f"Maximum Absolute Correlation: {self.best_max_abs_correlation:.6f}\n\n")

            f.write("Correlation Matrix:\n")
            for row in self.best_correlation_matrix:
                f.write(" ".join([f"{val:.6f}" for val in row]) + "\n")
            f.write("\n")

            for i, sequence in enumerate(self.best_sequences):
                f.write(f"Sequence {i + 1}: {sequence}\n\n")

    def verify_reproducibility(
            self,
            sequence_length=39,
            equal_probability=False,
            symbols=None,
            weights=None,
            verbose=True
    ):
        """Verify that the same seeds reproduce the same sequences and correlations."""
        if not self.best_seeds:
            raise ValueError("No seeds found yet. Run find_low_correlation_seeds first.")

        reproduced_sequences, reproduced_corr_matrix, reproduced_max_abs_corr = self.verify_sequences(
            self.best_seeds, sequence_length, equal_probability, symbols, weights
        )

        if verbose:
            print(f"Regenerated maximum absolute correlation: {reproduced_max_abs_corr:.6f}")

        # Check if sequences are identical to original ones
        sequences_match = all(
            orig_seq == repr_seq
            for orig_seq, repr_seq in zip(self.best_sequences, reproduced_sequences)
        )

        if verbose:
            print(f"All sequences reproduced correctly: {sequences_match}")

        return sequences_match, reproduced_max_abs_corr

    def get_best_results(self):
        """Get the best results found."""
        if not self.best_seeds:
            raise ValueError("No seeds found yet. Run find_low_correlation_seeds first.")

        return self.best_seeds, self.best_sequences, self.best_correlation_matrix, self.best_max_abs_correlation
