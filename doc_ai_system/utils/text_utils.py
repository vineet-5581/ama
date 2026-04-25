"""Text utility functions for normalization and analysis."""

import re
import unicodedata
from typing import List, Optional, Tuple


class TextUtils:
    """Utilities for text cleaning, normalization, and analysis."""

    # Common list bullet patterns
    BULLET_CHARS = set("•·‣◦▪▸-*–")
    NUMBERED_PATTERN = re.compile(
        r"^(\d+\.|[a-zA-Z]\.|[ivxlIVXL]+\.|\d+\))\s+"
    )

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove control characters and normalize unicode."""
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C") or ch in "\n\t")
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces/tabs into a single space."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def fix_broken_lines(lines: List[str]) -> List[str]:
        """Merge lines that appear broken mid-sentence."""
        if not lines:
            return lines

        result = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append("")
                continue

            if buffer:
                # Check if previous line ends without sentence terminator
                if buffer[-1] not in ".!?:;\"')" and not buffer[-1].isupper():
                    buffer = buffer + " " + line
                else:
                    result.append(buffer)
                    buffer = line
            else:
                buffer = line

        if buffer:
            result.append(buffer)

        return result

    @staticmethod
    def is_bullet_item(text: str) -> bool:
        """Check if text starts with a bullet character."""
        text = text.strip()
        if not text:
            return False
        first_char = text[0]
        return first_char in TextUtils.BULLET_CHARS

    @staticmethod
    def is_numbered_item(text: str) -> bool:
        """Check if text starts with a numbered list marker."""
        return bool(TextUtils.NUMBERED_PATTERN.match(text.strip()))

    @staticmethod
    def is_list_item(text: str) -> bool:
        """Check if text is a list item (bulleted or numbered)."""
        return TextUtils.is_bullet_item(text) or TextUtils.is_numbered_item(text)

    @staticmethod
    def strip_bullet(text: str) -> str:
        """Remove bullet character or number prefix from a list item."""
        text = text.strip()
        if TextUtils.is_bullet_item(text):
            return text[1:].strip()
        m = TextUtils.NUMBERED_PATTERN.match(text)
        if m:
            return text[m.end():].strip()
        return text

    @staticmethod
    def detect_encoding(text: str) -> str:
        """Heuristically detect the dominant script of text."""
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        has_arabic = any("\u0600" <= ch <= "\u06ff" for ch in text)
        has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in text)
        if has_cjk:
            return "cjk"
        if has_arabic:
            return "arabic"
        if has_cyrillic:
            return "cyrillic"
        return "latin"

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Simple sentence splitter using punctuation."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def is_heading_candidate(text: str, font_size: float, body_size: float) -> bool:
        """Check whether text is likely a heading based on font size ratio."""
        if not text.strip():
            return False
        ratio = font_size / max(body_size, 1.0)
        return ratio >= 1.2 and len(text.strip()) < 200

    @staticmethod
    def compute_text_similarity(a: str, b: str) -> float:
        """Compute a simple character-overlap similarity between two strings."""
        a, b = a.lower().strip(), b.lower().strip()
        if not a or not b:
            return 0.0
        set_a, set_b = set(a), set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to max_length characters."""
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def contains_url(text: str) -> bool:
        """Check whether text contains a URL."""
        url_pattern = re.compile(
            r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
        )
        return bool(url_pattern.search(text))

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract all URLs from text."""
        url_pattern = re.compile(
            r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
        )
        return url_pattern.findall(text)

    @staticmethod
    def merge_hyphenated_words(text: str) -> str:
        """Merge words broken by end-of-line hyphens."""
        return re.sub(r"-\n(\w)", r"\1", text)

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Normalize smart quotes to straight quotes."""
        replacements = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u00ab": '"',
            "\u00bb": '"',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text
