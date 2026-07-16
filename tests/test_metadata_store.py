from app.utils.metadata_store import MetadataStore


def test_list_documents_returns_all_ingested_documents_newest_first(tmp_path):
    store = MetadataStore(tmp_path / "metadata.db")

    store.upsert_document(
        doc_hash="hash-1",
        filename="first.pdf",
        page_count=1,
        chunk_count=2,
        image_count=0,
    )
    store.upsert_document(
        doc_hash="hash-2",
        filename="second.pdf",
        page_count=3,
        chunk_count=5,
        image_count=1,
    )

    documents = store.list_documents()

    assert [doc.hash for doc in documents] == ["hash-2", "hash-1"]
    assert documents[0].filename == "second.pdf"
    assert documents[0].chunk_count == 5


def test_list_documents_empty_when_nothing_ingested(tmp_path):
    store = MetadataStore(tmp_path / "metadata.db")
    assert store.list_documents() == []
