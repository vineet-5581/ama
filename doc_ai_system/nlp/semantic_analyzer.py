"""
Semantic analyzer for NLP-based document understanding.

Detects:
- Heading vs body text
- List items
- Section breaks
- Key-value pairs (form fields)
- Abstract / introduction / conclusion sections
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SemanticLabel:
    TITLE = "title"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    HEADING_4 = "heading_4"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    KEY_VALUE = "key_value"
    CODE = "code"
    ABSTRACT = "abstract"
    SECTION = "section"


class SemanticAnalyzer:
    """
    Assigns semantic labels to text blocks using rule-based heuristics
    with optional transformer model enhancement.

    The rule-based approach works well for the vast majority of documents.
    A BERT/LayoutLM model can be optionally loaded for higher accuracy.
    """

    # Common section headings (research paper, report)
    SECTION_KEYWORDS = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "methodology",
        "methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "references",
        "bibliography",
        "appendix",
        "acknowledgements",
        "acknowledgments",
        "summary",
        "overview",
        "objective",
        "objectives",
        "scope",
        "limitations",
        "future work",
    }

    def __init__(
        self,
        use_transformer: bool = False,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
    ):
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

        if use_transformer:
            self._load_transformer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text_blocks: List[Dict],
        body_font_size: Optional[float] = None,
    ) -> List[Dict]:
        """
        Add a 'semantic_label' field to each text block.

        Args:
            text_blocks: List of span/block dicts with 'text', 'size', 'flags'.
            body_font_size: Dominant body font size (auto-detected if None).

        Returns:
            Same list with 'semantic_label' added to each block.
        """
        if not text_blocks:
            return text_blocks

        if body_font_size is None:
            body_font_size = self._estimate_body_font_size(text_blocks)

        if self.use_transformer and self._model is not None:
            try:
                return self._analyze_transformer(text_blocks, body_font_size)
            except Exception as e:
                logger.warning(f"Transformer analysis failed, falling back: {e}")

        return self._analyze_rules(text_blocks, body_font_size)

    # ------------------------------------------------------------------
    # Rule-based analysis
    # ------------------------------------------------------------------

    def _analyze_rules(
        self, text_blocks: List[Dict], body_size: float
    ) -> List[Dict]:
        """Assign semantic labels using font-size heuristics and patterns."""
        labeled = []

        for block in text_blocks:
            text = block.get("text", "").strip()
            size = block.get("size", body_size)
            flags = block.get("flags", 0)

            label = self._classify_block(text, size, body_size, flags)
            annotated = dict(block)
            annotated["semantic_label"] = label
            labeled.append(annotated)

        return labeled

    def _classify_block(
        self, text: str, size: float, body_size: float, flags: int
    ) -> str:
        """Classify a single text block into a semantic label."""
        if not text.strip():
            return SemanticLabel.PARAGRAPH

        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        ratio = size / max(body_size, 1.0)

        is_bold = bool(flags & 0b10000)  # PyMuPDF bold flag
        is_italic = bool(flags & 0b01000)
        is_monospace = bool(flags & 0b1000000)

        # Code block detection
        if is_monospace or self._looks_like_code(text_stripped):
            return SemanticLabel.CODE

        # Title: very large, typically short
        if ratio >= 1.8 and len(text_stripped) < 200:
            return SemanticLabel.TITLE

        # Heading levels
        if ratio >= 1.5 and len(text_stripped) < 200:
            return SemanticLabel.HEADING_1
        if ratio >= 1.25 and len(text_stripped) < 200:
            return SemanticLabel.HEADING_2
        if ratio >= 1.1 and is_bold and len(text_stripped) < 200:
            return SemanticLabel.HEADING_3
        if is_bold and len(text_stripped) < 150:
            # Check if it looks like a section name
            if text_lower in self.SECTION_KEYWORDS:
                return SemanticLabel.HEADING_2
            return SemanticLabel.HEADING_4

        # Section keyword (same size as body but known section heading)
        if text_lower.rstrip(".") in self.SECTION_KEYWORDS:
            return SemanticLabel.SECTION

        # List item
        if self._is_list_item(text_stripped):
            return SemanticLabel.LIST_ITEM

        # Key-value pair (form fields, invoices)
        if self._is_key_value(text_stripped):
            return SemanticLabel.KEY_VALUE

        # Caption
        if self._is_caption(text_stripped, size, body_size):
            return SemanticLabel.CAPTION

        # Footnote (very small)
        if ratio < 0.8 and len(text_stripped) < 300:
            return SemanticLabel.FOOTNOTE

        return SemanticLabel.PARAGRAPH

    # ------------------------------------------------------------------
    # Pattern helpers
    # ------------------------------------------------------------------

    def _is_list_item(self, text: str) -> bool:
        bullet_chars = set("•·‣◦▪▸-*–")
        if text and text[0] in bullet_chars:
            return True
        return bool(
            re.match(r"^(\d+\.|[a-zA-Z]\.|[ivxlIVXL]+\.|\d+\))\s", text)
        )

    def _is_key_value(self, text: str) -> bool:
        """Detect 'Key: Value' or 'Key……Value' patterns."""
        return bool(
            re.match(r"^[A-Za-z][^:]{0,40}:\s*.+", text)
            or re.match(r"^[A-Za-z][^.]{0,40}\.{3,}\s*.+", text)
        )

    def _is_caption(self, text: str, size: float, body_size: float) -> bool:
        text_lower = text.lower().strip()
        is_small = size < body_size * 0.9
        starts_with_label = bool(
            re.match(
                r"^(figure|fig\.|table|tbl\.|chart|image|photo|illustration)\s*\d*",
                text_lower,
            )
        )
        return starts_with_label or (is_small and len(text) < 150)

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic: looks like source code."""
        code_indicators = [
            r"^\s*(def |class |import |from |#include|public |private |function )",
            r"[{};]$",
            r"^\s+\w+\(",
        ]
        for pattern in code_indicators:
            if re.search(pattern, text):
                return True
        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _estimate_body_font_size(self, text_blocks: List[Dict]) -> float:
        sizes = [b.get("size", 12.0) for b in text_blocks if b.get("size")]
        if not sizes:
            return 12.0
        from collections import Counter

        rounded = [round(s * 2) / 2 for s in sizes]
        return Counter(rounded).most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Transformer backend (optional)
    # ------------------------------------------------------------------

    def _load_transformer(self) -> None:
        """Attempt to load a BERT-like model for classification."""
        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info(f"Loaded transformer model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self.use_transformer = False

    def _analyze_transformer(
        self, text_blocks: List[Dict], body_size: float
    ) -> List[Dict]:
        """Use transformer model to enhance semantic labels."""
        # For now, run rules first and use transformer as a refinement signal.
        # Full implementation would fine-tune on document datasets.
        return self._analyze_rules(text_blocks, body_size)
