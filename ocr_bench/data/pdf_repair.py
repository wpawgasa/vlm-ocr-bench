"""Rebuild broken ToUnicode maps of digital statement PDFs (docparse task 1.2).

Some statements (KTB) embed a subset TrueType font as Type0/Identity-H with CID == GID
and a ToUnicode map that sends every code to itself, so text extraction yields glyph ids.
The embedded font still carries its own `cmap`, which gives the true character of each
glyph: the repair writes a copy of the PDF whose ToUnicode maps come from it.

A font is repaired only if its ToUnicode map is identity, the font's `cmap` resolves every
mapped code, and the result differs from the identity map. matplotlib-style PDFs (CID ==
code point, with a CIDToGIDMap stream) resolve to the identity map and are left alone.
Fonts with no `cmap` (Krungsri) have nothing to rebuild from and are left alone too.
"""

import io
import re
from pathlib import Path

import pikepdf
from fontTools.ttLib import TTFont

_BFCHAR_RE = re.compile(r"beginbfchar(.*?)endbfchar", re.S)
_BFRANGE_RE = re.compile(r"beginbfrange(.*?)endbfrange", re.S)
_HEX_PAIR_RE = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
_RANGE_RE = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]+>|\[[^\]]*\])")
_HEX_RE = re.compile(r"<([0-9A-Fa-f]+)>")
_BFCHAR_BLOCK = 100  # the CMap format's limit on entries per bfchar block


def _utf16(hex_str: str) -> str:
    return bytes.fromhex(hex_str).decode("utf-16-be", errors="replace")


def parse_to_unicode(data: bytes) -> dict[int, str]:
    """Map each code of a ToUnicode CMap to its text (bfchar and bfrange entries)."""
    text = data.decode("latin-1")
    mapping: dict[int, str] = {}
    for block in _BFCHAR_RE.findall(text):
        for src, dst in _HEX_PAIR_RE.findall(block):
            mapping[int(src, 16)] = _utf16(dst)
    for block in _BFRANGE_RE.findall(text):
        for lo, hi, dst in _RANGE_RE.findall(block):
            lo_i, hi_i = int(lo, 16), int(hi, 16)
            if dst.startswith("["):
                for code, item in zip(range(lo_i, hi_i + 1), _HEX_RE.findall(dst), strict=False):
                    mapping[code] = _utf16(item)
            else:
                base = bytes.fromhex(dst[1:-1])
                for offset in range(hi_i - lo_i + 1):
                    # the last byte of the destination increments across the range
                    value = base[:-1] + bytes([base[-1] + offset]) if base else base
                    mapping[lo_i + offset] = value.decode("utf-16-be", errors="replace")
    return mapping


def _to_unicode_stream(mapping: dict[int, str]) -> bytes:
    items = sorted(mapping.items())
    blocks = []
    for start in range(0, len(items), _BFCHAR_BLOCK):
        chunk = items[start : start + _BFCHAR_BLOCK]
        lines = "".join(
            f"<{code:04X}> <{text.encode('utf-16-be').hex().upper()}>\n" for code, text in chunk
        )
        blocks.append(f"{len(chunk)} beginbfchar\n{lines}endbfchar\n")
    return (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Repaired-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        + "".join(blocks)
        + "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode()


def _cid_to_gid(cid_font: pikepdf.Dictionary):
    """A function CID -> GID from the CIDToGIDMap (absent means Identity)."""
    c2g = cid_font.get("/CIDToGIDMap")
    if c2g is None or c2g == pikepdf.Name.Identity:
        return lambda cid: cid
    table = bytes(c2g.read_bytes())
    return lambda cid: (
        int.from_bytes(table[2 * cid : 2 * cid + 2], "big") if 2 * cid + 2 <= len(table) else None
    )


def _font_derived_map(type0: pikepdf.Dictionary, codes) -> dict[int, str] | None:
    """Each code's text from the embedded font's own `cmap`, or None if any code does not
    resolve (no embedded TrueType, no `cmap`, or a glyph without a character)."""
    if type0.get("/Encoding") != pikepdf.Name("/Identity-H"):
        return None
    descendants = type0.get("/DescendantFonts")
    if not descendants:
        return None
    cid_font = descendants[0]
    if cid_font.get("/Subtype") != pikepdf.Name.CIDFontType2:
        return None
    font_file = cid_font.get("/FontDescriptor", {}).get("/FontFile2")
    if font_file is None:
        return None
    font = TTFont(io.BytesIO(bytes(font_file.read_bytes())))
    if "cmap" not in font:
        return None
    gid_to_char: dict[int, int] = {}
    for code_point, glyph_name in font.getBestCmap().items():
        gid = font.getGlyphID(glyph_name)
        # a glyph reached from several code points (e.g. space and no-break space)
        # takes the lowest one
        gid_to_char[gid] = min(code_point, gid_to_char.get(gid, code_point))
    cid_to_gid = _cid_to_gid(cid_font)
    derived = {}
    for code in codes:
        gid = cid_to_gid(code)
        if gid not in gid_to_char:
            return None
        derived[code] = chr(gid_to_char[gid])
    return derived


def _fonts(resources, seen: set):
    """Every font reachable from `resources`, including those of Form XObjects; fonts and
    forms shared across pages are yielded and visited once."""
    if resources is None:
        return
    for font in (resources.get("/Font") or {}).values():
        key = font.objgen if font.is_indirect else id(font)
        if key not in seen:
            seen.add(key)
            yield font
    for xobject in (resources.get("/XObject") or {}).values():
        if xobject.get("/Subtype") == pikepdf.Name.Form and xobject.objgen not in seen:
            seen.add(xobject.objgen)
            yield from _fonts(xobject.get("/Resources"), seen)


def repair_tounicode(src: Path, dst: Path) -> int:
    """Write `src` to `dst` with every broken identity ToUnicode map rebuilt from its
    embedded font's `cmap`. Returns the number of fonts repaired; `dst` is written only
    if that number is above zero."""
    repaired = 0
    seen: set = set()
    with pikepdf.open(src) as pdf:
        for page in pdf.pages:
            for obj in _fonts(page.get("/Resources"), seen):
                repaired += _repair_font(pdf, obj)
        if repaired:
            pdf.save(dst)
    return repaired


def _repair_font(pdf: pikepdf.Pdf, obj: pikepdf.Dictionary) -> int:
    """Rebuild `obj`'s ToUnicode map in place if it is broken; 1 if repaired, else 0."""
    if obj.get("/Subtype") != pikepdf.Name.Type0:
        return 0
    to_unicode = obj.get("/ToUnicode")
    if not isinstance(to_unicode, pikepdf.Stream):
        return 0
    current = parse_to_unicode(bytes(to_unicode.read_bytes()))
    if not current or any(text != chr(code) for code, text in current.items()):
        return 0
    derived = _font_derived_map(obj, current)
    if derived is None or derived == current:
        return 0
    obj.ToUnicode = pdf.make_stream(_to_unicode_stream(derived))
    return 1
