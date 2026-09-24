"""Shared pytest fixtures: synthetic bank-statement files (no real corpus needed)."""

import io
from pathlib import Path

import matplotlib
import pikepdf
import pytest
from fontTools import subset
from fontTools.ttLib import TTFont
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


def _write_identity_tounicode_pdf(
    path, line: str = "KTB Statement Balance 1,000.00", keep_cmap: bool = True
) -> None:
    """A one-page PDF shaped like the real KTB statements: a subset TrueType font as
    Type0/Identity-H with CID == GID, whose ToUnicode maps every code to itself, so the
    extracted text is glyph ids. The embedded font keeps its `cmap` unless `keep_cmap` is
    False (the Krungsri shape: nothing left to rebuild the text from)."""
    font = TTFont(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    subsetter = subset.Subsetter(subset.Options())
    subsetter.populate(text=line)
    subsetter.subset(font)
    best = font.getBestCmap()
    gids = [font.getGlyphID(best[ord(ch)]) for ch in line]
    if not keep_cmap:
        del font["cmap"]
    buf = io.BytesIO()
    font.save(buf)

    pdf = pikepdf.new()
    descriptor = pikepdf.Dictionary(
        Type=pikepdf.Name.FontDescriptor,
        FontName=pikepdf.Name("/AAAAAB+Test"),
        Flags=32,
        FontBBox=[0, -250, 1000, 950],
        ItalicAngle=0,
        Ascent=950,
        Descent=-250,
        CapHeight=700,
        StemV=80,
        FontFile2=pdf.make_stream(buf.getvalue()),
    )
    cid_font = pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name.CIDFontType2,
        BaseFont=pikepdf.Name("/AAAAAB+Test"),
        CIDSystemInfo=pikepdf.Dictionary(
            Registry=pikepdf.String("Adobe"), Ordering=pikepdf.String("Identity"), Supplement=0
        ),
        FontDescriptor=pdf.make_indirect(descriptor),
        CIDToGIDMap=pikepdf.Name.Identity,
        DW=600,
    )
    entries = "".join(f"<{g:04X}> <{g:04X}>\n" for g in sorted(set(gids)))
    to_unicode = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        f"{len(set(gids))} beginbfchar\n{entries}endbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    type0 = pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name.Type0,
        BaseFont=pikepdf.Name("/AAAAAB+Test"),
        Encoding=pikepdf.Name("/Identity-H"),
        DescendantFonts=[pdf.make_indirect(cid_font)],
        ToUnicode=pdf.make_stream(to_unicode.encode()),
    )
    content = b"BT /F1 12 Tf 20 100 Td <" + "".join(f"{g:04X}" for g in gids).encode() + b"> Tj ET"
    pdf.pages.append(
        pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 400, 200],
                Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=type0)),
                Contents=pdf.make_stream(content),
            )
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path)


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


@pytest.fixture
def statements_dir_with_identity_tounicode(statements_dir):
    """`statements_dir` plus a KTB-shaped digital PDF whose text layer is recoverable
    from the embedded font's own `cmap`."""
    _write_identity_tounicode_pdf(statements_dir / "ktb_identity.pdf")
    return statements_dir
