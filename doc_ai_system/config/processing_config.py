"""Processing configuration for the Document AI System."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing pipeline."""

    # Page rendering
    page_dpi: int = 150
    page_scale: float = 2.0

    # Parallel processing
    max_workers: int = 4
    use_parallel: bool = True

    # OCR settings
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_confidence_threshold: float = 60.0

    # Text extraction
    min_font_size: float = 4.0
    max_font_size: float = 120.0

    # Layout analysis
    column_gap_threshold: float = 20.0
    min_block_width: float = 10.0
    min_block_height: float = 5.0

    # Table detection
    table_extraction_enabled: bool = True
    table_min_rows: int = 2
    table_min_cols: int = 2

    # Image extraction
    image_extraction_enabled: bool = True
    min_image_width: int = 50
    min_image_height: int = 50

    # Caching
    cache_enabled: bool = True
    cache_dir: str = ".cache"

    # Output
    output_dir: str = "output"

    # Supported document types
    supported_types: List[str] = field(
        default_factory=lambda: [
            "resume",
            "research_paper",
            "invoice",
            "form",
            "book_page",
            "generic",
        ]
    )

    # Post-processing
    fix_broken_lines: bool = True
    merge_fragments: bool = True
    normalize_whitespace: bool = True

    # Advanced options
    use_layout_model: bool = False
    use_nlp_model: bool = False
    layoutlm_model: Optional[str] = None
    bert_model: Optional[str] = "bert-base-uncased"
