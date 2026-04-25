"""Style configuration for Word document generation."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class HeadingThreshold:
    """Font-size thresholds for heading level detection."""

    h1_min: float = 20.0
    h2_min: float = 16.0
    h3_min: float = 14.0
    h4_min: float = 12.0
    body_min: float = 8.0


@dataclass
class StyleConfig:
    """Configuration for document style reconstruction."""

    # Heading detection
    heading_thresholds: HeadingThreshold = field(
        default_factory=HeadingThreshold
    )

    # Font mapping
    default_font: str = "Calibri"
    default_font_size: float = 11.0
    heading_font: str = "Calibri"
    code_font: str = "Courier New"

    # Paragraph spacing
    paragraph_space_before: float = 6.0
    paragraph_space_after: float = 6.0
    line_spacing: float = 1.15

    # Heading spacing
    heading_space_before: float = 12.0
    heading_space_after: float = 6.0

    # Page margins (inches)
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0

    # Table styles
    table_style: str = "Table Grid"
    table_font_size: float = 10.0

    # List styles
    bullet_indent: float = 0.25
    list_item_space: float = 2.0

    # Color preservation
    preserve_text_color: bool = True
    preserve_background_color: bool = False

    # Bold/italic/underline detection
    bold_weight_threshold: float = 600.0
    detect_italic: bool = True
    detect_underline: bool = True

    # Heading color mapping (RGB tuples)
    heading_colors: Dict[int, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            1: (26, 26, 26),
            2: (38, 38, 38),
            3: (51, 51, 51),
            4: (64, 64, 64),
        }
    )

    # Known list bullet characters
    bullet_chars: List[str] = field(
        default_factory=lambda: [
            "•",
            "·",
            "‣",
            "◦",
            "▪",
            "▸",
            "-",
            "*",
            "–",
        ]
    )

    # Numbered list patterns
    numbered_list_patterns: List[str] = field(
        default_factory=lambda: [
            r"^\d+\.",
            r"^\d+\)",
            r"^[a-zA-Z]\.",
            r"^[ivxlIVXL]+\.",
        ]
    )
