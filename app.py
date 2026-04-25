"""
Main application entry point
Orchestrates the full Document AI Pipeline
"""
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging first
from utils.logging_utils import setup_logging
logger = setup_logging("DocumentAI")

from pipeline.base import Document, DocumentProcessingPipeline
from pipeline.classifier import DocumentClassifier
from pipeline.text_extractor import TextExtractor
from cv.layout_detector import LayoutDetector
from nlp.semantic_analyzer import SemanticAnalyzer
from tables.extractor import TableExtractor
from generator.word_generator import WordGeneratorComponent
from utils.file_utils import ensure_output_directory, get_file_hash


class DocumentAIPipeline:
    """
    Complete document processing pipeline
    """

    def __init__(self, use_ml: bool = False, use_ocr: bool = False, use_transformers: bool = False):
        self.use_ml = use_ml
        self.use_ocr = use_ocr
        self.use_transformers = use_transformers
        self.pipeline = None

    def _build_pipeline(self) -> DocumentProcessingPipeline:
        """Build the complete processing pipeline"""
        pipeline = DocumentProcessingPipeline()

        # Add components in order
        logger.info("Building processing pipeline...")

        # Layer 1: Classification
        pipeline.add_component(DocumentClassifier())
        logger.debug("Added DocumentClassifier")

        # Layer 4: Text Extraction
        pipeline.add_component(TextExtractor())
        logger.debug("Added TextExtractor")

        # Layer 6: Table Extraction
        pipeline.add_component(TableExtractor(use_pdfplumber=True))
        logger.debug("Added TableExtractor")

        # Layer 7: Semantic Analysis
        pipeline.add_component(SemanticAnalyzer(use_transformers=self.use_transformers))
        logger.debug("Added SemanticAnalyzer")

        # Layer 2: Layout Detection (optional)
        if self.use_ml:
            pipeline.add_component(LayoutDetector(use_ml=True))
            logger.debug("Added LayoutDetector")

        return pipeline

    def process_pdf(self, pdf_path: str, output_path: str) -> str:
        """
        Process a PDF and generate Word document
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path to output DOCX file
            
        Returns:
            Path to generated Word document
        """
        logger.info(f"Starting PDF processing: {pdf_path}")
        logger.info(f"Output: {output_path}")

        # Validate inputs
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        ensure_output_directory(output_path)

        # Create document object
        document = Document(
            title="Converted Document",
            metadata={"pdf_path": str(Path(pdf_path).absolute())}
        )

        # Build pipeline if not already built
        if not self.pipeline:
            self.pipeline = self._build_pipeline()

        try:
            # Execute pipeline
            logger.info("Executing processing pipeline...")
            document = self.pipeline.process(document)

            # Generate Word document
            logger.info("Generating Word document...")
            generator = WordGeneratorComponent(
                output_path=output_path,
                preserve_colors=True,
                preserve_images=True
            )
            result_path = generator.generate(document)

            logger.info(f"Successfully generated: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise

    def process_batch(self, pdf_directory: str, output_directory: str):
        """
        Process multiple PDFs in a directory
        
        Args:
            pdf_directory: Directory containing PDFs
            output_directory: Directory for output DOCX files
        """
        pdf_dir = Path(pdf_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        successful = 0
        failed = 0

        for pdf_file in pdf_files:
            try:
                output_path = output_dir / pdf_file.stem / ".docx"
                self.process_pdf(str(pdf_file), str(output_path))
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                failed += 1

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Document AI Platform - Convert PDF to Word"
    )
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("output", help="Output DOCX file or directory")
    parser.add_argument(
        "--ml", action="store_true", help="Use ML models for layout detection"
    )
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for scanned documents")
    parser.add_argument(
        "--transformers",
        action="store_true",
        help="Use transformer models for semantic analysis",
    )
    parser.add_argument("--batch", action="store_true", help="Process directory of PDFs")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = DocumentAIPipeline(
        use_ml=args.ml,
        use_ocr=args.ocr,
        use_transformers=args.transformers,
    )

    try:
        if args.batch:
            pipeline.process_batch(args.input, args.output)
        else:
            result = pipeline.process_pdf(args.input, args.output)
            print(f"\nSuccess! Generated: {result}")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
