"""Image utility functions for preprocessing and analysis."""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utilities for image processing using OpenCV and numpy."""

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert an RGB/RGBA image to grayscale."""
        import cv2

        if img.ndim == 2:
            return img
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    @staticmethod
    def binarize(img: np.ndarray, method: str = "otsu") -> np.ndarray:
        """Binarize an image using Otsu's or adaptive thresholding."""
        import cv2

        gray = ImageUtils.to_grayscale(img)
        if method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
        else:
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
        """Apply non-local means denoising to reduce noise."""
        import cv2

        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(img, h=strength)
        return cv2.fastNlMeansDenoisingColored(img, h=strength)

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Deskew an image by detecting and correcting rotation angle."""
        import cv2

        gray = ImageUtils.to_grayscale(img)
        # Invert colors so text is white on black
        inverted = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inverted > 0))
        if len(coords) == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angles in [-90, 0)
        if angle < -45:
            angle = 90 + angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        import cv2

        gray = ImageUtils.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def remove_shadows(img: np.ndarray) -> np.ndarray:
        """Remove background shadow using morphological operations."""
        import cv2

        gray = ImageUtils.to_grayscale(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        diff = 255 - cv2.absdiff(gray, background)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return norm

    @staticmethod
    def resize(
        img: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Resize an image by target dimensions or scale factor."""
        import cv2

        h, w = img.shape[:2]
        if scale is not None:
            new_w = int(w * scale)
            new_h = int(h * scale)
        elif width is not None and height is not None:
            new_w, new_h = width, height
        elif width is not None:
            new_w = width
            new_h = int(h * width / w)
        elif height is not None:
            new_h = height
            new_w = int(w * height / h)
        else:
            return img

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def crop(
        img: np.ndarray, x0: int, y0: int, x1: int, y1: int
    ) -> np.ndarray:
        """Crop a region from an image."""
        return img[y0:y1, x0:x1]

    @staticmethod
    def numpy_to_pil(img: np.ndarray):
        """Convert a numpy array (RGB) to a PIL Image."""
        from PIL import Image

        if img.ndim == 2:
            return Image.fromarray(img, mode="L")
        return Image.fromarray(img, mode="RGB")

    @staticmethod
    def pil_to_numpy(pil_img) -> np.ndarray:
        """Convert a PIL Image to a numpy array."""
        return np.array(pil_img)

    @staticmethod
    def encode_to_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
        """Encode a numpy image array to bytes."""
        import cv2
        import io
        from PIL import Image

        pil = ImageUtils.numpy_to_pil(img)
        buf = io.BytesIO()
        pil.save(buf, format=fmt)
        return buf.getvalue()

    @staticmethod
    def get_dimensions(img: np.ndarray) -> Tuple[int, int]:
        """Return (width, height) of the image."""
        h, w = img.shape[:2]
        return w, h
