"""PDF utility functions for opening, rendering, and inspecting PDFs."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PDFUtils:
    """Utilities for working with PDF files using PyMuPDF."""

    @staticmethod
    def open_pdf(path: str):
        """Open a PDF file and return a fitz.Document."""
        import fitz  # PyMuPDF

        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(path)
        logger.info(f"Opened PDF: {path} ({doc.page_count} pages)")
        return doc

    @staticmethod
    def get_page_count(path: str) -> int:
        """Return the number of pages in a PDF."""
        import fitz

        with fitz.open(path) as doc:
            return doc.page_count

    @staticmethod
    def render_page_to_image(page, dpi: int = 150, scale: float = 2.0):
        """Render a PDF page to a PIL Image."""
        import fitz
        from PIL import Image
        import io

        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        return img

    @staticmethod
    def render_page_to_numpy(page, dpi: int = 150, scale: float = 2.0):
        """Render a PDF page to a numpy array (BGR for OpenCV)."""
        import fitz
        import numpy as np

        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        # Convert to numpy array (RGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            # RGBA → RGB
            arr = arr[:, :, :3]
        return arr

    @staticmethod
    def extract_text_blocks(page) -> List[Dict]:
        """Extract text blocks with position and style info from a page."""
        blocks = []
        raw_blocks = page.get_text("rawdict")["blocks"]

        for block in raw_blocks:
            if block.get("type") != 0:  # type 0 = text
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    bbox = span.get("bbox", (0, 0, 0, 0))
                    blocks.append(
                        {
                            "text": text,
                            "bbox": bbox,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "font": span.get("font", ""),
                            "size": span.get("size", 12.0),
                            "flags": span.get("flags", 0),
                            "color": span.get("color", 0),
                            "origin": span.get("origin", (bbox[0], bbox[1])),
                            "block_no": block.get("number", 0),
                            "line_no": line.get("wmode", 0),
                        }
                    )

        return blocks

    @staticmethod
    def extract_text_dict(page) -> Dict:
        """Return the full text dictionary for a page."""
        return page.get_text("dict")

    @staticmethod
    def extract_images(page, doc) -> List[Dict]:
        """Extract images embedded in a PDF page."""
        images = []
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                images.append(
                    {
                        "xref": xref,
                        "ext": base_image["ext"],
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "image": base_image["image"],
                        "bbox": page.get_image_bbox(img_info),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not extract image xref={xref}: {e}")

        return images

    @staticmethod
    def extract_links(page) -> List[Dict]:
        """Extract hyperlinks from a PDF page."""
        links = []
        for link in page.get_links():
            link_type = link.get("kind", 0)
            uri = link.get("uri", "")
            rect = link.get("from", None)
            if uri:
                links.append(
                    {
                        "uri": uri,
                        "bbox": (
                            (rect.x0, rect.y0, rect.x1, rect.y1) if rect else None
                        ),
                        "kind": link_type,
                    }
                )
        return links

    @staticmethod
    def is_scanned_pdf(page, text_threshold: int = 50) -> bool:
        """Heuristically detect if a page is scanned (image-based)."""
        text = page.get_text("text").strip()
        return len(text) < text_threshold

    @staticmethod
    def get_page_dimensions(page) -> Tuple[float, float]:
        """Return the (width, height) of a page in points."""
        rect = page.rect
        return rect.width, rect.height

    @staticmethod
    def get_metadata(doc) -> Dict:
        """Extract metadata from the PDF document."""
        meta = doc.metadata or {}
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "page_count": doc.page_count,
        }
