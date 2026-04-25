"""Validation configuration for quality metrics."""

from dataclasses import dataclass


@dataclass
class ValidationConfig:
    """Configuration for output validation and quality metrics."""

    # Layout similarity thresholds
    layout_similarity_threshold: float = 0.80
    layout_warning_threshold: float = 0.70

    # Text accuracy thresholds
    text_accuracy_threshold: float = 0.95
    text_warning_threshold: float = 0.85

    # Table accuracy thresholds
    table_accuracy_threshold: float = 0.90
    table_warning_threshold: float = 0.75

    # Completeness checks
    min_text_coverage: float = 0.85
    max_lost_blocks: int = 5

    # Post-processing quality gates
    enable_quality_checks: bool = True
    fail_on_low_quality: bool = False

    # Reporting
    generate_report: bool = False
    report_path: str = "output/quality_report.json"

    # Retry on failure
    max_retries: int = 2
    retry_with_ocr: bool = True
