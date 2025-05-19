from typing import Dict

import matplotlib.pyplot as plt
import numpy as np, os

class Visualization:
    @staticmethod
    def visualize_watermark_comparison(
            original: np.ndarray,
            recovered: np.ndarray,
            image_name: str,
            gain_factor: float,
            watermark_type: int = 1,
            save_path: str = "watermark_comparison.tif",
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
        plt.savefig(save_path)
        plt.close()


    @staticmethod
    def plot_relationship(
            x_values,
            results,
            title,
            xlabel,
            ylabel,
            save_path,
            ylim=None,
            legend_loc='best'
    ):
        """Helper method to create standardized plots"""
        plt.figure(figsize=(10, 6))
        markers = ['s', 'o', '*', '+', 'x', 'd', '^']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

        for i, (label, values) in enumerate(results.items()):
            plt.plot(
                x_values,
                values,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                label=label
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend(loc=legend_loc)

        if ylim is not None:
            plt.ylim(ylim)

        plt.savefig(save_path)
        plt.show()

        return results
