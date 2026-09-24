"""TDD tests for ocr_bench.data.bankstmt (tasks 2.2, 2.3)."""

import numpy as np
import pytest

from ocr_bench.config import BankStmtConfig, CorpusTargets
from ocr_bench.data.bankstmt import (
    UnknownBankError,
    bank_for,
    coverage_warnings,
    discover,
    prepare_statements,
    rasterize_page,
    route,
    text_layer,
    text_layer_usable,
)
from ocr_bench.schemas import NoGT, TextLayerGT
from tests.conftest import _write_scanned_pdf

BANKS = ["kbank", "scb", "bbl", "ktb", "krungsri", "ttb", "gsb"]
BANK_MAP = {
    "*kbank*": "kbank",
    "*scb*": "scb",
    "*bbl*": "bbl",
    "*ktb*": "ktb",
    "*krungsri*": "krungsri",
    "*ttb*": "ttb",
    "*gsb*": "gsb",
}


def _cfg(input_dir) -> BankStmtConfig:
    return BankStmtConfig(
        kind="bankstmt",
        input_dir=str(input_dir),
        dpi=200,
        banks=BANKS,
        bank_map=BANK_MAP,
    )


class TestDiscover:
    def test_finds_all_fixture_files_sorted(self, statements_dir):
        files = discover(statements_dir)
        rels = [p.relative_to(statements_dir).as_posix() for p in files]
        assert rels == sorted(rels)
        assert "bbl_photo.png" in rels
        assert "kbank_0001.pdf" in rels
        assert "scb_scan.pdf" in rels

    def test_ignores_non_document_files(self, statements_dir, tmp_path):
        (statements_dir / "notes.txt").write_text("hi")
        files = discover(statements_dir)
        rels = {p.name for p in files}
        assert "notes.txt" not in rels


class TestBankFor:
    def test_known_bank_matches(self):
        cfg = _cfg("/nonexistent")
        assert bank_for("stmts/kbank_0001.pdf", cfg) == "kbank"
        assert bank_for("stmts/scb_scan.pdf", cfg) == "scb"

    def test_unknown_bank_raises(self):
        cfg = _cfg("/nonexistent")
        with pytest.raises(UnknownBankError) as exc_info:
            bank_for("misc/unknown.pdf", cfg)
        assert "misc/unknown.pdf" in str(exc_info.value)

    def test_first_matching_pattern_wins(self):
        cfg = BankStmtConfig(
            kind="bankstmt",
            input_dir="/nonexistent",
            banks=["kbank", "scb"],
            bank_map={"*kbank*": "kbank", "*kbank_0001*": "scb"},
        )
        assert bank_for("stmts/kbank_0001.pdf", cfg) == "kbank"


class TestRoute:
    def test_digital_pdf_routes_digital(self, statements_dir):
        assert route(statements_dir / "kbank_0001.pdf") == "digital"

    def test_scanned_pdf_routes_scanned(self, statements_dir):
        assert route(statements_dir / "scb_scan.pdf") == "scanned"


class TestRasterizeAndTextLayer:
    def test_rasterize_page_size_matches_dpi(self, statements_dir):
        img = rasterize_page(statements_dir / "kbank_0001.pdf", 1, 200)
        h, w = img.shape[:2]
        # 4in x 3in page at 200 dpi -> 800 x 600, allow +/-2px for rounding.
        assert abs(w - 800) <= 2
        assert abs(h - 600) <= 2

    def test_text_layer_contains_balance_and_boxes_within_image(self, statements_dir):
        pdf = statements_dir / "kbank_0001.pdf"
        img = rasterize_page(pdf, 1, 200)
        h, w = img.shape[:2]
        gt = text_layer(pdf, 1, 200)
        assert isinstance(gt, TextLayerGT)
        assert "Balance" in gt.gt_text
        assert len(gt.gt_words) > 0
        for word in gt.gt_words:
            x0, y0, x1, y1 = word.bbox
            assert -1 <= x0 <= x1 <= w + 1
            assert -1 <= y0 <= y1 <= h + 1


class TestPrepareStatements:
    def test_missing_input_dir_raises(self, tmp_path):
        cfg = _cfg(tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError):
            prepare_statements(cfg)

    def test_sample_ids_and_counts(self, statements_dir):
        cfg = _cfg(statements_dir)
        samples, info = prepare_statements(cfg)
        ids = {s.sample_id for s in samples}
        assert "BS-kbank-0001-p1" in ids
        assert "BS-kbank-0001-p2" in ids
        assert "BS-scb-0001-p1" in ids
        assert "BS-bbl-0001-p1" in ids
        assert len(samples) == 4  # 2 kbank pages + 1 scb + 1 bbl

    def test_digital_pages_get_text_layer_gt(self, statements_dir):
        cfg = _cfg(statements_dir)
        samples, _ = prepare_statements(cfg)
        kbank_p1 = next(s for s in samples if s.sample_id == "BS-kbank-0001-p1")
        assert isinstance(kbank_p1.gt, TextLayerGT)
        assert kbank_p1.doc_type == "digital"
        assert kbank_p1.n_pages == 2
        assert kbank_p1.bank == "kbank"
        assert kbank_p1.domain == "Finance"
        assert kbank_p1.critical_fields == [
            "account_no",
            "opening_balance",
            "closing_balance",
            "period_start",
            "period_end",
        ]
        assert isinstance(kbank_p1.image, np.ndarray)

    def test_scanned_pages_get_no_gt(self, statements_dir):
        cfg = _cfg(statements_dir)
        samples, _ = prepare_statements(cfg)
        scb = next(s for s in samples if s.sample_id == "BS-scb-0001-p1")
        assert isinstance(scb.gt, NoGT)
        assert scb.doc_type == "scanned"

    def test_photo_gets_no_gt_and_doc_type_photo(self, statements_dir):
        cfg = _cfg(statements_dir)
        samples, _ = prepare_statements(cfg)
        bbl = next(s for s in samples if s.sample_id == "BS-bbl-0001-p1")
        assert isinstance(bbl.gt, NoGT)
        assert bbl.doc_type == "photo"
        assert bbl.n_pages == 1

    def test_unknown_bank_raises(self, statements_dir_with_unknown_bank):
        cfg = _cfg(statements_dir_with_unknown_bank)
        with pytest.raises(UnknownBankError):
            prepare_statements(cfg)

    def test_unknown_banks_fail_fast_listing_all(
        self, statements_dir_with_unknown_bank, monkeypatch
    ):
        # Bank resolution must happen for every file before any rasterization,
        # and the error must name every unresolvable file at once.
        import ocr_bench.data.bankstmt as bankstmt

        _write_scanned_pdf(statements_dir_with_unknown_bank / "zzz" / "other.pdf")

        def _no_raster(*a, **k):
            raise AssertionError("rasterized before bank validation")

        monkeypatch.setattr(bankstmt, "rasterize_page", _no_raster)
        cfg = _cfg(statements_dir_with_unknown_bank)
        with pytest.raises(UnknownBankError) as exc_info:
            prepare_statements(cfg)
        msg = str(exc_info.value)
        assert "misc/unknown.pdf" in msg
        assert "zzz/other.pdf" in msg

    def test_summary_info(self, statements_dir):
        cfg = _cfg(statements_dir)
        _, info = prepare_statements(cfg)
        assert info["statement_pages_by_doc_type"]["digital"] == 2
        assert info["statement_pages_by_doc_type"]["scanned"] == 1
        assert info["statement_pages_by_doc_type"]["photo"] == 1
        assert info["statement_banks"] == ["bbl", "kbank", "scb"]
        assert info["statement_files"] == 3
        assert info["statement_multipage_files"] == 1


class TestTextLayerUsable:
    def test_english_keywords_usable(self):
        assert text_layer_usable("Account No. 123 Balance 1,000.00")

    def test_thai_keywords_usable(self):
        assert text_layer_usable("ยอดยกมา 1,000.00 คงเหลือ")

    def test_scrambled_glyph_codes_unusable(self):
        # Real KTB text layer (custom glyph encoding with a wrong ToUnicode map)
        assert not text_layer_usable(',BBMSLR=R@RDKDLR =R@RDKDLR:DPHMC ! !" "!RM#!')

    def test_empty_unusable(self):
        assert not text_layer_usable("   ")

    def test_garbled_digital_page_gets_no_gt_but_stays_digital(
        self, statements_dir_with_garbled_text_layer
    ):
        samples, info = prepare_statements(_cfg(statements_dir_with_garbled_text_layer))
        ktb = next(s for s in samples if s.bank == "ktb")
        assert ktb.doc_type == "digital"
        assert isinstance(ktb.gt, NoGT)
        kbank = next(s for s in samples if s.sample_id == "BS-kbank-0001-p1")
        assert isinstance(kbank.gt, TextLayerGT)
        assert info["statement_text_layer_unusable"] == 1

    def test_identity_tounicode_page_is_repaired_into_gt(
        self, statements_dir_with_identity_tounicode
    ):
        samples, info = prepare_statements(_cfg(statements_dir_with_identity_tounicode))
        ktb = next(s for s in samples if s.bank == "ktb")
        assert isinstance(ktb.gt, TextLayerGT)
        assert "Statement Balance 1,000.00" in ktb.gt.gt_text
        assert any(w.text == "Balance" for w in ktb.gt.gt_words)
        assert info["statement_text_layer_unusable"] == 0
        assert info["statement_text_layer_repaired"] == 1


class TestCoverageWarnings:
    def test_digital_target_counts_only_usable_text_layers(self):
        info = {
            "statement_pages_by_doc_type": {"digital": 50, "scanned": 50},
            "statement_text_layer_unusable": 20,
            "statement_banks": ["a", "b", "c", "d"],
            "statement_multipage_files": 10,
        }
        warnings = coverage_warnings(info, CorpusTargets())
        assert len(warnings) == 1
        assert "only 30 digital pages with a usable text layer" in warnings[0]

    def test_under_target_corpus_yields_all_four_warnings(self, statements_dir):
        cfg = _cfg(statements_dir)
        _, info = prepare_statements(cfg)
        targets = CorpusTargets(
            min_digital_pages=40,
            min_scanned_photo_pages=40,
            min_banks=4,
            min_multipage_files=10,
        )
        warnings = coverage_warnings(info, targets)
        assert len(warnings) == 4
        assert any("digital pages" in w for w in warnings)
        assert any("scanned/photo pages" in w for w in warnings)
        assert any("banks" in w for w in warnings)
        assert any("multi-page" in w for w in warnings)

    def test_met_target_yields_no_warnings(self, statements_dir):
        cfg = _cfg(statements_dir)
        _, info = prepare_statements(cfg)
        targets = CorpusTargets(
            min_digital_pages=0,
            min_scanned_photo_pages=0,
            min_banks=0,
            min_multipage_files=0,
        )
        assert coverage_warnings(info, targets) == []
