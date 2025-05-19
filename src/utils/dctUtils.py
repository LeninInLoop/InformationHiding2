import numpy as np
from scipy.fft import dctn, idctn
from skimage.util import view_as_blocks


class DCTUtils:
    @staticmethod
    def dctn(array: np.ndarray, axes: tuple, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return np.array(dctn(array, axes=axes, type=dct_type, norm=norm), dtype=np.float64)

    @staticmethod
    def idctn(array: np.ndarray, axes: tuple, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return np.array(idctn(array, axes=axes, type=dct_type, norm=norm), dtype=np.float64)

    @staticmethod
    def block_dct(image: np.ndarray, block_shape: int = 8) -> np.ndarray:
        blocks = view_as_blocks(image, block_shape=block_shape)
        return DCTUtils.dctn(blocks, axes=(2, 3))

    @staticmethod
    def block_idct(dct_blocks: np.ndarray) -> np.ndarray:
        recon_blocks = DCTUtils.idctn(dct_blocks, axes=(2, 3), norm='ortho')
        rows = [np.concatenate(row_blocks, axis=1) for row_blocks in recon_blocks]
        return np.concatenate(rows, axis=0)

    @staticmethod
    def extract_dc_values(dct_blocks: np.ndarray) -> np.ndarray:
        return dct_blocks[:, :, 0, 0]

    @staticmethod
    def replace_dc_values(dct_blocks: np.ndarray, dc_values: np.ndarray) -> np.ndarray:
        result = dct_blocks.copy()
        h_blocks, w_blocks = dc_values.shape
        for i in range(h_blocks):
            for j in range(w_blocks):
                result[i, j, 0, 0] = dc_values[i, j]
        return result