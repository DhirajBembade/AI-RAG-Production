from pathlib import Path

from app.services.pdf_extractor import extract_pdf

SAMPLE_PDF = (
    Path(__file__).resolve().parent.parent / "data" / "attention_is_all_you_need.pdf"
)


def test_extract_pdf_returns_text_and_images(tmp_path):
    pages = extract_pdf(SAMPLE_PDF, images_dir=tmp_path, doc_hash="testdoc")

    assert len(pages) > 0
    assert any(page.text for page in pages)

    total_images = sum(len(page.images) for page in pages)
    assert total_images > 0

    for page in pages:
        for image in page.images:
            assert image.path.exists()
            assert image.source in {"embedded", "page_render"}
