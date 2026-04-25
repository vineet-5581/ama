"""Image preprocessing pipeline for OCR enhancement."""

import logging
from typing import Optional

import numpy as np

from ..utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Prepares scanned or low-quality page images for OCR.

    Applies a configurable sequence of operations:
    - Grayscale conversion
    - Shadow removal
    - Denoising
    - Deskewing
    - Contrast enhancement
    - Binarization
    """

    def __init__(
        self,
        denoise: bool = True,
        deskew: bool = True,
        binarize: bool = True,
        remove_shadows: bool = True,
        enhance_contrast: bool = True,
        denoise_strength: int = 10,
        binarize_method: str = "adaptive",
    ):
        self.denoise = denoise
        self.deskew = deskew
        self.binarize = binarize
        self.remove_shadows = remove_shadows
        self.enhance_contrast = enhance_contrast
        self.denoise_strength = denoise_strength
        self.binarize_method = binarize_method

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing pipeline on an image.

        Args:
            img: Input image as a numpy array (RGB or grayscale).

        Returns:
            Preprocessed image as a numpy array.
        """
        try:
            # Step 1: Convert to grayscale
            processed = ImageUtils.to_grayscale(img)

            # Step 2: Remove shadows / uneven illumination
            if self.remove_shadows:
                processed = ImageUtils.remove_shadows(processed)

            # Step 3: Denoise
            if self.denoise:
                processed = ImageUtils.denoise(
                    processed, strength=self.denoise_strength
                )

            # Step 4: Enhance contrast
            if self.enhance_contrast:
                processed = ImageUtils.enhance_contrast(processed)

            # Step 5: Deskew
            if self.deskew:
                processed = ImageUtils.deskew(processed)

            # Step 6: Binarize
            if self.binarize:
                processed = ImageUtils.binarize(
                    processed, method=self.binarize_method
                )

            return processed

        except Exception as e:
            logger.warning(f"Preprocessing failed, returning original: {e}")
            return ImageUtils.to_grayscale(img)

    def preprocess_for_table_detection(self, img: np.ndarray) -> np.ndarray:
        """
        Lighter preprocessing optimized for table detection.
        Preserves structural lines better than full OCR preprocessing.
        """
        try:
            import cv2

            gray = ImageUtils.to_grayscale(img)
            # Apply Gaussian blur to reduce noise while preserving lines
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # Threshold
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            return binary
        except Exception as e:
            logger.warning(f"Table preprocessing failed: {e}")
            return ImageUtils.to_grayscale(img)

    def preprocess_for_layout(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocessing optimized for layout detection.
        Returns an enhanced color or grayscale image.
        """
        try:
            import cv2

            if img.ndim == 3 and img.shape[2] == 3:
                # Return lightly enhanced color image
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                enhanced_lab = cv2.merge([l_channel, a, b])
                return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            return ImageUtils.enhance_contrast(img)
        except Exception as e:
            logger.warning(f"Layout preprocessing failed: {e}")
            return img
