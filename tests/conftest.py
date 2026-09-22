"""Shared pytest fixtures: synthetic bank-statement files (no real corpus needed)."""

import matplotlib
import pytest
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42


def _write_digital_pdf(
    path, n_pages: int = 2, line: str = "KBANK Statement Balance 1,000.00"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(path)) as pdf:
        for i in range(n_pages):
            fig = plt.figure(figsize=(4, 3))
            fig.text(0.05, 0.5, line, fontsize=10)
            if "Balance" in line:
                fig.text(0.05, 0.3, f"Page {i + 1}", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def _write_scanned_pdf(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    img.save(str(path), "PDF")


def _write_photo_png(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 150), color=(120, 130, 140))
    img.save(str(path), "PNG")


@pytest.fixture
def statements_dir(tmp_path):
    """A small synthetic bank-statement corpus: 1 digital PDF, 1 scanned PDF, 1 photo."""
    stmts = tmp_path / "stmts"
    _write_digital_pdf(stmts / "kbank_0001.pdf", n_pages=2)
    _write_scanned_pdf(stmts / "scb_scan.pdf")
    _write_photo_png(stmts / "bbl_photo.png")
    return stmts


@pytest.fixture
def statements_dir_with_unknown_bank(statements_dir):
    """`statements_dir` plus a file whose name matches no configured bank pattern."""
    _write_scanned_pdf(statements_dir / "misc" / "unknown.pdf")
    return statements_dir


@pytest.fixture
def statements_dir_with_garbled_text_layer(statements_dir):
    """`statements_dir` plus a digital PDF whose text layer is scrambled glyph codes
    (as seen in real KTB/Krungsri statements with broken font encodings)."""
    _write_digital_pdf(statements_dir / "ktb_garbled.pdf", n_pages=1, line="0HHSXRW ,BBMSLR=R@RDK")
    return statements_dir
