"""Tests for shared ingestion staging helpers."""

import zipfile

import pytest

from nof1_causal_lab.actions.ingestion.tools import _safe_resolve

pytestmark = pytest.mark.contract


class TestSafeResolve:
    def test_normal_path(self, tmp_path):
        child = tmp_path / "data.csv"
        child.touch()
        assert _safe_resolve(tmp_path, "data.csv") == child.resolve()

    def test_nested_path(self, tmp_path):
        nested = tmp_path / "sub"
        nested.mkdir()
        child = nested / "data.csv"
        child.touch()
        assert _safe_resolve(tmp_path, "sub/data.csv") == child.resolve()

    def test_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal blocked"):
            _safe_resolve(tmp_path, "../../../etc/passwd")

    def test_sibling_prefix_traversal_blocked(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        sibling = tmp_path / "base_evil"
        sibling.write_text("outside")

        with pytest.raises(ValueError, match="Path traversal blocked"):
            _safe_resolve(base, "../base_evil")


class TestPrepareRawInput:
    def test_extracts_zip_archives(self, tmp_path):
        from nof1_causal_lab.actions.ingestion.flow import _prepare_raw_input

        raw_zip = tmp_path / "input.zip"
        with zipfile.ZipFile(raw_zip, "w") as zf:
            zf.writestr("nested/data.csv", "date,value\n2024-01-01,1\n")

        prepared_dir = tmp_path / "prepared"
        result = _prepare_raw_input(raw_zip, prepared_dir)

        assert result == prepared_dir
        assert (prepared_dir / "nested" / "data.csv").read_text() == "date,value\n2024-01-01,1\n"

    def test_copies_non_archive_files(self, tmp_path):
        from nof1_causal_lab.actions.ingestion.flow import _prepare_raw_input

        raw_text = tmp_path / "input.txt"
        raw_text.write_text("line one\nline two\n")

        prepared_dir = tmp_path / "prepared"
        result = _prepare_raw_input(raw_text, prepared_dir)

        assert result == prepared_dir
        assert (prepared_dir / "input.txt").read_text() == "line one\nline two\n"
