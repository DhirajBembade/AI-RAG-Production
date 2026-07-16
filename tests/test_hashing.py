import hashlib

from app.services.hashing import sha256_bytes, sha256_file, sha256_text


def test_sha256_text_is_deterministic():
    assert sha256_text("hello world") == sha256_text("hello world")


def test_sha256_text_differs_for_different_input():
    assert sha256_text("hello") != sha256_text("world")


def test_sha256_bytes_matches_hashlib():
    assert sha256_bytes(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_sha256_text_matches_utf8_encoded_bytes():
    assert sha256_text("hello") == sha256_bytes("hello".encode("utf-8"))


def test_sha256_file_matches_bytes_hash(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"some file content for hashing")
    assert sha256_file(str(file_path)) == sha256_bytes(b"some file content for hashing")
