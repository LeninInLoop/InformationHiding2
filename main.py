import os
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dctn, idctn
from PIL import Image


class BColors:
    HEADER = '\033[95m'
    OkBLUE = '\033[94m'
    OkCYAN = '\033[96m'
    OkGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError
        img = np.array(Image.open(path), dtype=np.float32)
        return img

    @staticmethod
    def normalize(img: np.ndarray, dtype: str) -> np.ndarray:
        max_img = np.max(img)
        min_img = np.min(img)

        normalized_image = (img - min_img) / (max_img - min_img) * 255
        if dtype == 'uint8':
            return normalized_image.astype(np.uint8)
        else:
            return normalized_image.astype(np.float32)

    @staticmethod
    def save_image(path: str, img: np.ndarray):
        if img.dtype != np.uint8 and img.dtype != np.bool:
            img = ImageUtils.normalize(img, dtype='uint8')
        return Image.fromarray(img).save(path)

    @staticmethod
    def convert_to_gray_scale(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize(image, dtype='uint8')
        return np.array(Image.fromarray(image).convert('L'))

    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize(image, dtype='uint8')
        img = Image.fromarray(image)
        resized_img = img.resize(target_size)
        return np.array(resized_img)


class DCTUtils:
    @staticmethod
    def dctn(array: np.ndarray, dct_type: int, norm: str) -> np.ndarray:
        return dctn(array, type=dct_type, norm=norm)

    @staticmethod
    def idctn(array: np.ndarray, dct_type: int, norm: str) -> np.ndarray:
        return idctn(array, type=dct_type, norm=norm)

    @staticmethod
    def block_dct(image: np.ndarray, block_size: int = 8) -> np.ndarray:
        height, width = image.shape

        h_blocks = height // block_size
        w_blocks = width // block_size

        dct_blocks = np.zeros(
            (h_blocks, w_blocks, block_size, block_size),
            dtype=np.float32
        )
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = image[
                        i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size
                        ]
                dct_block = DCTUtils.dctn(block, dct_type=2, norm='ortho')
                dct_blocks[i, j] = dct_block
        return dct_blocks

    @staticmethod
    def extract_dc_values(dct_blocks: np.ndarray) -> np.ndarray:
        return dct_blocks[:, :, 0, 0]

    @staticmethod
    def create_low_res_approx_image(image: np.ndarray) -> np.ndarray:
        dct_blocks = DCTUtils.block_dct(
            image=image,
            block_size=8
        )
        dc_values = DCTUtils.extract_dc_values(dct_blocks)
        return dc_values


class WatermarkUtils:
    @staticmethod
    def create_watermark_image(save_path: str) -> np.ndarray:
        watermark_array = np.array(
            [
                [0, 1, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 0, 1, 1, 0, 1, 0],
            ]
        ).astype(np.bool)
        ImageUtils.save_image(
            img=watermark_array,
            path=save_path,
        )
        return watermark_array

    @staticmethod
    def generate_noise(shape, n_streams=1, seed=None):
        rng = np.random.default_rng(seed)
        choices = [-1, 0, 1]
        probs = [1 / 3, 1 / 3, 1 / 3]

        out_shape = (n_streams, *shape) if n_streams > 1 else shape
        flat_size = int(np.prod(out_shape))

        flat = rng.choice(choices, size=flat_size, p=probs)
        noise = flat.reshape(out_shape)
        return noise


class Helper:
    @staticmethod
    def create_directories(directories: Dict) -> None:
        os.makedirs(directories['image_base_path'], exist_ok=True)
        os.makedirs(directories["original_gray_path"], exist_ok=True)
        os.makedirs(directories["low_res_approx_image_path"], exist_ok=True)
        os.makedirs(directories["watermark_path"], exist_ok=True)


def main():



    directories = {
        "image_base_path": "Images",
        "original_gray_path": "Images/Original Gray Scale",
        "low_res_approx_image_path": "Images/Low Res Approx",
        "watermark_path": "Images/Watermark"
    }
    Helper.create_directories(directories)




    image = "goldhill"  # goldhill
    original_path = os.path.join(directories['image_base_path'], f"{image}.bmp")
    original_image = ImageUtils.load_image(original_path)
    print("Original Image Size: ", original_image.shape)





    original_gray_scale = ImageUtils.convert_to_gray_scale(original_image)
    ImageUtils.save_image(
        path=os.path.join(directories["original_gray_path"], f"{image}_gray_scale.bmp"),
        img=original_gray_scale
    )
    print("Original Gray Scale Image Size: ", original_gray_scale.shape)






    if original_gray_scale.shape[0] != 512 or original_gray_scale.shape[1] != 512:

        original_gray_scale = ImageUtils.resize_image(
            image=original_gray_scale,
            target_size=(512, 512)
        )

        ImageUtils.save_image(
            img=original_gray_scale,
            path=os.path.join(directories["original_gray_path"], f"{image}_gray_scale_resized.bmp")
        )

        print("Resized Image Size: ", original_gray_scale.shape)






    approx_image = DCTUtils.create_low_res_approx_image(original_gray_scale)
    ImageUtils.save_image(
        path=os.path.join(directories["low_res_approx_image_path"], f"{image}_approx_image.bmp"),
        img=approx_image
    )
    print("Low Res Approx Image Size: ", approx_image.shape)





    bdct_of_low_res_approx_image = DCTUtils.block_dct(
        image=approx_image,
        block_size=8
    )
    print("BDCT of LRAI Size", bdct_of_low_res_approx_image.shape)




    watermark = WatermarkUtils.create_watermark_image(
        save_path=os.path.join(directories["watermark_path"], f"watermark.bmp")
    )
    print("Watermark Size: ", watermark.shape)



    # Option 1 to generate Noise
    noise1 = WatermarkUtils.generate_noise(
        shape=(1, 39),
        n_streams=1,
        seed=44
    )
    noise2 = WatermarkUtils.generate_noise(
        shape=(1, 39),
        n_streams=1,
        seed=48
    )
    noise = np.array([noise1, noise2])
    print("Noise Size: ", noise.shape)
    print("Noise: \n", noise)
    print("Covariance of Generated Noises:\n", np.cov(noise[0], noise[1]))



    # Option 2 to generate Noise
    noise = WatermarkUtils.generate_noise(
        shape=(1, 39),
        n_streams=2,
        seed=47
    )
    print("Noise Size: ", noise.shape)
    print("Noise: \n", noise)
    print("Covariance of Generated Noises:\n", np.cov(noise[0], noise[1]))




if __name__ == '__main__':
    main()

