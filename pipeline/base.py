"""
Base data structures and abstract classes for the document pipeline
Core domain models and component interfaces
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    RESUME = "resume"
    RESEARCH_PAPER = "research_paper"
    INVOICE = "invoice"
    FORM = "form"
    BOOK_PAGE = "book_page"
    LETTER = "letter"
    ARTICLE = "article"
    UNKNOWN = "unknown"


class TextAlignment(Enum):
    """Text alignment options"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class TableBorderType(Enum):
    """Table border types"""
    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    DOTTED = "dotted"
    DASHED = "dashed"


@dataclass
class BoundingBox:
    """Bounding box for spatial information (in points, 72 DPI)"""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Calculate width"""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Calculate height"""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Calculate area"""
        return self.width * self.height

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


@dataclass
class TextStyle:
    """Text styling information"""
    font_name: str = "Calibri"
    font_size: float = 11.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color_hex: Optional[str] = None  # e.g., "#000000"
    bg_color_hex: Optional[str] = None
    text_alignment: TextAlignment = TextAlignment.LEFT
    line_spacing: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "font_name": self.font_name,
            "font_size": self.font_size,
            "bold": self.bold,
            "italic": self.italic,
            "underline": self.underline,
            "strikethrough": self.strikethrough,
            "color_hex": self.color_hex,
            "bg_color_hex": self.bg_color_hex,
            "text_alignment": self.text_alignment.value,
            "line_spacing": self.line_spacing,
        }


@dataclass
class Hyperlink:
    """Hyperlink information"""
    text: str
    url: str
    title: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "url": self.url,
            "title": self.title,
        }


@dataclass
class TextBlock:
    """Text block with styling and spatial information"""
    text: str
    style: TextStyle = field(default_factory=TextStyle)
    bbox: Optional[BoundingBox] = None
    page_num: int = 0
    confidence: float = 1.0
    hyperlinks: List[Hyperlink] = field(default_factory=list)
    is_heading: bool = False
    heading_level: int = 0  # 1-6 for h1-h6
    is_list_item: bool = False
    list_level: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "style": self.style.to_dict(),
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "page_num": self.page_num,
            "confidence": self.confidence,
            "hyperlinks": [h.to_dict() for h in self.hyperlinks],
            "is_heading": self.is_heading,
            "heading_level": self.heading_level,
            "is_list_item": self.is_list_item,
            "list_level": self.list_level,
        }


@dataclass
class TableCell:
    """Table cell with content and properties"""
    content: str
    row_span: int = 1
    col_span: int = 1
    bbox: Optional[BoundingBox] = None
    style: TextStyle = field(default_factory=TextStyle)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "style": self.style.to_dict(),
        }


@dataclass
class TableRow:
    """Table row containing cells"""
    cells: List[TableCell] = field(default_factory=list)
    is_header: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "is_header": self.is_header,
        }


@dataclass
class Table:
    """Table structure with rows and columns"""
    rows: List[TableRow] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    page_num: int = 0
    title: Optional[str] = None
    border_type: TableBorderType = TableBorderType.SINGLE

    @property
    def num_rows(self) -> int:
        """Get number of rows"""
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        """Get number of columns"""
        if not self.rows:
            return 0
        return len(self.rows[0].cells)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "rows": [row.to_dict() for row in self.rows],
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "page_num": self.page_num,
            "title": self.title,
            "border_type": self.border_type.value,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
        }


@dataclass
class Figure:
    """Figure/Image with metadata"""
    image_path: str  # Local or embedded
    bbox: Optional[BoundingBox] = None
    page_num: int = 0
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    width: Optional[float] = None  # in inches
    height: Optional[float] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "image_path": self.image_path,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "page_num": self.page_num,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


@dataclass
class DocumentPage:
    """Single page in a document"""
    page_num: int
    width: float  # in points
    height: float
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    rotation: int = 0  # 0, 90, 180, 270
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "text_blocks": [tb.to_dict() for tb in self.text_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "rotation": self.rotation,
        }


@dataclass
class Document:
    """Complete document with metadata and content"""
    title: str = "Untitled"
    pages: List[DocumentPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 1.0
    language: str = "en"

    # Processing pipeline metadata
    processing_steps: List[str] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)

    @property
    def num_pages(self) -> int:
        """Get total number of pages"""
        return len(self.pages)

    @property
    def total_tables(self) -> int:
        """Get total number of tables"""
        return sum(len(page.tables) for page in self.pages)

    @property
    def total_figures(self) -> int:
        """Get total number of figures"""
        return sum(len(page.figures) for page in self.pages)

    def add_page(self, page: DocumentPage):
        """Add page to document"""
        self.pages.append(page)
        logger.debug(f"Added page {page.page_num} to document")

    def get_all_text_blocks(self) -> List[TextBlock]:
        """Get all text blocks from all pages"""
        return [tb for page in self.pages for tb in page.text_blocks]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "num_pages": self.num_pages,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
            "doc_type": self.doc_type.value,
            "confidence": self.confidence,
            "language": self.language,
            "total_tables": self.total_tables,
            "total_figures": self.total_figures,
            "processing_steps": self.processing_steps,
            "processing_errors": self.processing_errors,
        }


class PipelineComponent(ABC):
    """Abstract base class for pipeline components"""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    @abstractmethod
    def process(self, document: Document) -> Document:
        """
        Process document
        
        Args:
            document: Document to process
            
        Returns:
            Processed document
        """
        pass

    def validate_input(self, document: Document) -> bool:
        """
        Validate input document
        
        Args:
            document: Document to validate
            
        Returns:
            True if valid
        """
        return document is not None and len(document.pages) > 0

    def __call__(self, document: Document) -> Document:
        """Make component callable"""
        try:
            self.logger.info(f"Processing with {self.name}")
            if not self.validate_input(document):
                self.logger.warning(f"{self.name}: Invalid input, returning as-is")
                return document
            result = self.process(document)
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}", exc_info=True)
            document.processing_errors.append(f"{self.name}: {str(e)}")
            return document


class DocumentProcessingPipeline:
    """Pipeline for sequential document processing"""

    def __init__(self):
        self.components: List[PipelineComponent] = []
        self.logger = logging.getLogger("pipeline.DocumentProcessingPipeline")

    def add_component(self, component: PipelineComponent):
        """Add processing component to pipeline"""
        self.components.append(component)
        self.logger.info(f"Added component: {component.name}")

    def process(self, document: Document) -> Document:
        """
        Process document through all components in order
        
        Args:
            document: Document to process
            
        Returns:
            Processed document
        """
        self.logger.info(f"Starting pipeline processing for '{document.title}'")

        for component in self.components:
            self.logger.debug(f"Executing component: {component.name}")
            document = component(document)
            document.processing_steps.append(component.name)

        self.logger.info(
            f"Pipeline complete. Processed {document.num_pages} pages, "
            f"{document.total_tables} tables, {document.total_figures} figures"
        )

        return document

    def get_component(self, name: str) -> Optional[PipelineComponent]:
        """Get component by name"""
        for component in self.components:
            if component.name == name:
                return component
        return None

    def __repr__(self) -> str:
        """String representation"""
        components_str = " -> ".join([c.name for c in self.components])
        return f"DocumentProcessingPipeline({components_str})"
