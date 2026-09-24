import subprocess

from ocr_bench.data.bankstmt import text_layer_usable
from ocr_bench.data.pdf_repair import repair_tounicode
from tests.conftest import _write_digital_pdf, _write_identity_tounicode_pdf

LINE = "KTB Statement Balance 1,000.00"


def _pdftotext(path) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True
    ).stdout


class TestRepairToUnicode:
    def test_identity_map_yields_glyph_ids_before_repair(self, tmp_path):
        src = tmp_path / "ktb.pdf"
        _write_identity_tounicode_pdf(src, LINE)
        assert not text_layer_usable(_pdftotext(src))

    def test_repair_rebuilds_text_from_embedded_cmap(self, tmp_path):
        src, dst = tmp_path / "ktb.pdf", tmp_path / "ktb.repaired.pdf"
        _write_identity_tounicode_pdf(src, LINE)
        assert repair_tounicode(src, dst) == 1
        assert LINE in _pdftotext(dst)

    def test_correct_identity_map_is_left_alone(self, tmp_path):
        # matplotlib uses code points as CIDs, so its identity ToUnicode is right
        src, dst = tmp_path / "mpl.pdf", tmp_path / "mpl.repaired.pdf"
        _write_digital_pdf(src, n_pages=1)
        assert repair_tounicode(src, dst) == 0
        assert not dst.exists()

    def test_font_without_cmap_is_not_repaired(self, tmp_path):
        # Krungsri shape: no `cmap` in the subset font, so nothing to rebuild from
        src, dst = tmp_path / "bay.pdf", tmp_path / "bay.repaired.pdf"
        _write_identity_tounicode_pdf(src, LINE, keep_cmap=False)
        assert repair_tounicode(src, dst) == 0
        assert not dst.exists()
