from typing import Tuple

from PIL import Image
import numpy as np, os

class ImageUtils:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        return np.array(Image.open(path), dtype=np.float32)

    @staticmethod
    def normalize(img: np.ndarray, dtype: str) -> np.ndarray:
        max_img = np.max(img)
        min_img = np.min(img)
        if max_img == min_img:
            normalized_image = np.full_like(img, 127.5)
        else:
            normalized_image = (img - min_img) / (max_img - min_img) * 255
        return normalized_image.astype(np.uint8 if dtype == 'uint8' else np.float32)

    @staticmethod
    def save_image(path: str, img: np.ndarray) -> None:
        if img.dtype != np.uint8 and img.dtype != np.bool:
            img = ImageUtils.normalize(img, dtype='uint8')
        Image.fromarray(img).save(path)

    @staticmethod
    def convert_to_gray_scale(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize(image, dtype='uint8')
        return np.array(Image.fromarray(image).convert('L'))

    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize(image, dtype='uint8')
        return np.array(Image.fromarray(image).resize(target_size))

