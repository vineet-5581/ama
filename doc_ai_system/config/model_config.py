"""Model configuration for AI/ML components."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for AI/ML models used in the pipeline."""

    # Document Classifier
    classifier_model: str = "rules"  # "rules" | "ml" | "transformer"
    classifier_threshold: float = 0.6

    # Layout Detection
    layout_model: str = "heuristic"  # "heuristic" | "layoutlmv3" | "detectron2"
    layout_confidence: float = 0.5
    layout_model_path: Optional[str] = None

    # OCR Engine
    ocr_engine: str = "tesseract"  # "tesseract" | "trocr" | "easyocr"
    tesseract_path: Optional[str] = None
    tesseract_config: str = "--oem 3 --psm 6"

    # Table Detection
    table_model: str = "pdfplumber"  # "pdfplumber" | "tablenet" | "hybrid"
    table_confidence: float = 0.5

    # NLP / Semantic Analysis
    nlp_model: str = "rules"  # "rules" | "bert" | "layoutlm"
    nlp_model_name: str = "bert-base-uncased"
    nlp_max_length: int = 512

    # Reading Order
    reading_order: str = "spatial"  # "spatial" | "ml"

    # Device settings
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    use_half_precision: bool = False

    # Model caching
    model_cache_dir: str = ".model_cache"

    # Batch processing
    batch_size: int = 8

    # Supported languages for OCR
    ocr_languages: List[str] = field(
        default_factory=lambda: ["eng", "fra", "deu", "spa", "ita"]
    )
