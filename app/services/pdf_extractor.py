import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    path: Path
    page_number: int
    source: str  # "embedded" | "page_render"


@dataclass
class PageContent:
    page_number: int
    text: str
    images: list[ExtractedImage] = field(default_factory=list)
    ocr_text: str = ""


def extract_pdf(
    pdf_path: str | Path,
    images_dir: Path,
    doc_hash: str,
    min_text_chars_for_page_render: int = 40,
    ocr_enabled: bool = True,
) -> list[PageContent]:
    """Hybrid extraction with three complementary sources per page:

    1. Native PDF text layer (page.get_text()) — fast, exact, works for normal PDFs.
    2. Embedded raster images, pulled out directly (page.get_images).
    3. For pages whose native text is sparse (scanned pages, vector-drawn diagrams with
       no embedded raster image): a full-page render, which is both (a) OCR'd via
       pytesseract to recover any literal text a scan contains, and (b) captioned by a
       vision model later in the pipeline for a semantic description of diagrams/figures.
    """
    doc_images_dir = images_dir / doc_hash
    doc_images_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageContent] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            text = page.get_text().strip()
            images: list[ExtractedImage] = []
            ocr_text = ""

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace and pix.colorspace.n > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    out_path = doc_images_dir / f"page{page_number}_img{img_index}.png"
                    pix.save(str(out_path))
                    images.append(
                        ExtractedImage(
                            path=out_path, page_number=page_number, source="embedded"
                        )
                    )
                except Exception:
                    logger.warning(
                        "Skipping unreadable embedded image on page %d (xref=%s)",
                        page_number,
                        xref,
                    )

            if len(text) < min_text_chars_for_page_render:
                pix = page.get_pixmap(dpi=150)
                out_path = doc_images_dir / f"page{page_number}_render.png"
                pix.save(str(out_path))
                images.append(
                    ExtractedImage(
                        path=out_path, page_number=page_number, source="page_render"
                    )
                )

                if ocr_enabled:
                    try:
                        ocr_text = pytesseract.image_to_string(
                            Image.open(out_path)
                        ).strip()
                    except Exception as exc:
                        # Missing/broken tesseract binary shouldn't break ingestion —
                        # OCR is a bonus signal, not a hard requirement.
                        logger.warning(
                            "OCR failed for page %d (%s); continuing without it",
                            page_number,
                            exc,
                        )

            pages.append(
                PageContent(
                    page_number=page_number, text=text, images=images, ocr_text=ocr_text
                )
            )

    return pages
