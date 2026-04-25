"""OCR engine supporting Tesseract and fallback strategies."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from ..utils.image_utils import ImageUtils
from .preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class OCRResult:
    """Container for OCR output."""

    def __init__(self, text: str, confidence: float, blocks: List[Dict]):
        self.text = text
        self.confidence = confidence
        self.blocks = blocks  # List of {text, bbox, confidence}

    def __repr__(self) -> str:
        return f"OCRResult(confidence={self.confidence:.1f}%, chars={len(self.text)})"


class OCREngine:
    """
    Multi-backend OCR engine.

    Supports:
    - Tesseract (default, robust, multi-language)
    - EasyOCR (optional, deep-learning based)

    Falls back gracefully when engines are unavailable.
    """

    def __init__(
        self,
        engine: str = "tesseract",
        lang: str = "eng",
        tesseract_config: str = "--oem 3 --psm 6",
        confidence_threshold: float = 60.0,
        preprocess: bool = True,
        tesseract_path: Optional[str] = None,
    ):
        self.engine = engine
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.confidence_threshold = confidence_threshold
        self.preprocess = preprocess
        self.preprocessor = ImagePreprocessor()
        self._easyocr_reader = None

        if tesseract_path:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            except ImportError:
                pass

    def run(self, img: np.ndarray) -> OCRResult:
        """
        Run OCR on an image.

        Args:
            img: Input image as numpy array.

        Returns:
            OCRResult with extracted text, confidence, and block details.
        """
        if self.engine == "easyocr":
            return self._run_easyocr(img)
        return self._run_tesseract(img)

    def _run_tesseract(self, img: np.ndarray) -> OCRResult:
        """Run Tesseract OCR on the image."""
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract not installed, returning empty OCR result")
            return OCRResult("", 0.0, [])

        # Preprocess
        if self.preprocess:
            processed = self.preprocessor.preprocess(img)
        else:
            processed = ImageUtils.to_grayscale(img)

        pil_img = ImageUtils.numpy_to_pil(processed)

        try:
            # Get detailed data for confidence and bounding boxes
            data = pytesseract.image_to_data(
                pil_img,
                lang=self.lang,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return OCRResult("", 0.0, [])

        blocks = []
        texts = []
        confidences = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            conf = float(data["conf"][i])
            word = data["text"][i].strip()
            if conf < 0 or not word:
                continue

            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            blocks.append(
                {
                    "text": word,
                    "bbox": (x, y, x + w, y + h),
                    "confidence": conf,
                }
            )
            if conf >= self.confidence_threshold:
                texts.append(word)
                confidences.append(conf)

        full_text = " ".join(texts)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return OCRResult(full_text, avg_confidence, blocks)

    def _run_easyocr(self, img: np.ndarray) -> OCRResult:
        """Run EasyOCR on the image."""
        try:
            import easyocr
        except ImportError:
            logger.warning("easyocr not installed, falling back to tesseract")
            return self._run_tesseract(img)

        if self._easyocr_reader is None:
            logger.info("Initializing EasyOCR reader...")
            self._easyocr_reader = easyocr.Reader([self.lang])

        try:
            results = self._easyocr_reader.readtext(img)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return self._run_tesseract(img)

        blocks = []
        texts = []
        confidences = []

        for result in results:
            bbox_pts, text, conf = result
            conf_pct = conf * 100
            flat_bbox = (
                int(bbox_pts[0][0]),
                int(bbox_pts[0][1]),
                int(bbox_pts[2][0]),
                int(bbox_pts[2][1]),
            )
            blocks.append(
                {"text": text, "bbox": flat_bbox, "confidence": conf_pct}
            )
            if conf_pct >= self.confidence_threshold:
                texts.append(text)
                confidences.append(conf_pct)

        full_text = " ".join(texts)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return OCRResult(full_text, avg_confidence, blocks)

    def run_on_region(
        self,
        img: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> OCRResult:
        """Run OCR on a specific region of the image."""
        region = ImageUtils.crop(img, x0, y0, x1, y1)
        return self.run(region)

    def is_available(self) -> bool:
        """Check whether the configured OCR engine is available."""
        if self.engine == "tesseract":
            try:
                import pytesseract

                pytesseract.get_tesseract_version()
                return True
            except Exception:
                return False
        elif self.engine == "easyocr":
            try:
                import easyocr

                return True
            except ImportError:
                return False
        return False
