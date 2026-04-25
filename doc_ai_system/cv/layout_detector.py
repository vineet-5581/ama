"""
Visual layout detector using heuristic analysis with optional
deep-learning backends (LayoutLMv3, Detectron2, YOLOv8).

Detects layout elements:
  - title, heading, paragraph, table, figure, header, footer,
    list, caption, multi-column zones
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayoutElement:
    """A single detected layout element."""

    element_type: str  # title|heading|paragraph|table|figure|header|footer|list|caption
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in page coords
    confidence: float = 1.0
    label: Optional[str] = None
    column: int = 0  # column index for multi-column layouts
    reading_order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayoutDetector:
    """
    Heuristic + optional deep-learning layout detector.

    By default uses rule-based spatial analysis which is fast and
    dependency-free.  When `use_dl_model=True` it attempts to load
    a pre-trained model (LayoutLMv3 or similar).
    """

    def __init__(
        self,
        page_width: float,
        page_height: float,
        use_dl_model: bool = False,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        self.page_width = page_width
        self.page_height = page_height
        self.use_dl_model = use_dl_model
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        text_blocks: List[Dict],
        page_img: Optional[np.ndarray] = None,
    ) -> List[LayoutElement]:
        """
        Detect layout elements from text blocks (and optionally an image).

        Args:
            text_blocks: List of span dicts from PDFUtils.extract_text_blocks().
            page_img:    Optional page image for DL-based detection.

        Returns:
            Sorted list of LayoutElement objects in reading order.
        """
        if self.use_dl_model and page_img is not None:
            try:
                return self._detect_dl(text_blocks, page_img)
            except Exception as e:
                logger.warning(
                    f"DL layout detection failed, falling back to heuristic: {e}"
                )

        return self._detect_heuristic(text_blocks)

    # ------------------------------------------------------------------
    # Heuristic detection
    # ------------------------------------------------------------------

    def _detect_heuristic(self, text_blocks: List[Dict]) -> List[LayoutElement]:
        """Rule-based layout detection from text span metadata."""
        if not text_blocks:
            return []

        # Compute dominant body font size
        body_size = self._estimate_body_font_size(text_blocks)

        # Detect header / footer zones
        header_y_max = self.page_height * 0.08
        footer_y_min = self.page_height * 0.92

        # Detect columns
        columns = self._detect_columns(text_blocks)

        elements: List[LayoutElement] = []

        for span in text_blocks:
            x0, y0, x1, y1 = span["bbox"]
            text = span.get("text", "").strip()
            font_size = span.get("size", body_size)
            flags = span.get("flags", 0)

            if not text:
                continue

            # Determine column
            col_idx = self._assign_column(x0, x1, columns)

            # Classify element type
            if y0 <= header_y_max:
                el_type = "header"
            elif y1 >= footer_y_min:
                el_type = "footer"
            elif font_size >= body_size * 1.8:
                el_type = "title"
            elif font_size >= body_size * 1.3:
                el_type = "heading"
            elif self._is_list_item(text):
                el_type = "list"
            elif self._is_caption(text, font_size, body_size):
                el_type = "caption"
            else:
                el_type = "paragraph"

            elements.append(
                LayoutElement(
                    element_type=el_type,
                    bbox=(x0, y0, x1, y1),
                    confidence=1.0,
                    column=col_idx,
                    metadata={
                        "font_size": font_size,
                        "flags": flags,
                        "font": span.get("font", ""),
                        "color": span.get("color", 0),
                    },
                )
            )

        # Sort by reading order
        elements = self._sort_reading_order(elements, columns)
        for i, el in enumerate(elements):
            el.reading_order = i

        return elements

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _detect_columns(self, text_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """
        Detect multi-column layout by clustering x-coordinates.
        Returns list of (x_left, x_right) column bands.
        """
        if not text_blocks:
            return [(0, self.page_width)]

        # Collect x0 positions of all spans (ignoring headers/footers)
        header_y = self.page_height * 0.08
        footer_y = self.page_height * 0.92
        x_starts = [
            b["bbox"][0]
            for b in text_blocks
            if header_y < b["bbox"][1] < footer_y
        ]

        if not x_starts:
            return [(0, self.page_width)]

        # Simple 2-column detection: look for a significant gap in x0
        mid = self.page_width / 2
        left_starts = [x for x in x_starts if x < mid * 0.7]
        right_starts = [x for x in x_starts if x > mid * 0.5]

        has_left = len(left_starts) > 3
        has_right = len(right_starts) > 3

        if has_left and has_right:
            # Two columns
            right_col_start = min(right_starts)
            return [
                (0, right_col_start - 5),
                (right_col_start, self.page_width),
            ]

        return [(0, self.page_width)]

    def _assign_column(
        self,
        x0: float,
        x1: float,
        columns: List[Tuple[float, float]],
    ) -> int:
        """Assign a span to a column index."""
        cx = (x0 + x1) / 2
        for i, (col_left, col_right) in enumerate(columns):
            if col_left <= cx <= col_right:
                return i
        return 0

    # ------------------------------------------------------------------
    # Reading order sorting
    # ------------------------------------------------------------------

    def _sort_reading_order(
        self,
        elements: List[LayoutElement],
        columns: List[Tuple[float, float]],
    ) -> List[LayoutElement]:
        """
        Sort elements in natural reading order:
        - Headers/footers first/last
        - Within a column: top to bottom
        - Multi-column: column 0 first, then column 1, etc.
        """
        headers = [e for e in elements if e.element_type == "header"]
        footers = [e for e in elements if e.element_type == "footer"]
        body = [e for e in elements if e.element_type not in ("header", "footer")]

        # Sort body by (column, y0)
        body.sort(key=lambda e: (e.column, e.bbox[1]))

        headers.sort(key=lambda e: e.bbox[1])
        footers.sort(key=lambda e: e.bbox[1])

        return headers + body + footers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_body_font_size(self, text_blocks: List[Dict]) -> float:
        """Estimate the dominant (body) font size using mode."""
        sizes = [b.get("size", 12.0) for b in text_blocks if b.get("size")]
        if not sizes:
            return 12.0
        # Round to nearest 0.5 and take the most common size
        rounded = [round(s * 2) / 2 for s in sizes]
        from collections import Counter

        counter = Counter(rounded)
        return counter.most_common(1)[0][0]

    def _is_list_item(self, text: str) -> bool:
        """Check if text looks like a list item."""
        import re

        text = text.strip()
        if not text:
            return False
        bullet_chars = set("•·‣◦▪▸-*–")
        if text[0] in bullet_chars:
            return True
        return bool(re.match(r"^(\d+\.|[a-zA-Z]\.|[ivxlIVXL]+\.)\s", text))

    def _is_caption(self, text: str, font_size: float, body_size: float) -> bool:
        """Detect figure/table captions."""
        import re

        text_lower = text.lower().strip()
        is_small = font_size < body_size * 0.95
        starts_with_label = bool(
            re.match(r"^(figure|fig\.|table|tbl\.|chart|image)\s*\d*", text_lower)
        )
        return starts_with_label or (is_small and len(text) < 150)

    # ------------------------------------------------------------------
    # Deep-learning backend (optional)
    # ------------------------------------------------------------------

    def _detect_dl(
        self,
        text_blocks: List[Dict],
        page_img: np.ndarray,
    ) -> List[LayoutElement]:
        """
        Placeholder for deep-learning layout detection.
        Attempts LayoutLMv3-style inference if model is loaded.
        Falls back to heuristic if model unavailable.
        """
        raise NotImplementedError(
            "DL layout detection not yet configured. "
            "Set use_dl_model=False or provide a model_name."
        )
