"""Text canonicalization shared by the prediction and ground-truth sides (task 4.1).

`canonical_text` is the spec's "Text canonicalization"; `plain_text` is the markup-free
view both sides are reduced to before CER/WER. Neither mutates its input, so callers keep
the raw string alongside the normalized one.
"""

import html
import re
import unicodedata

# Legacy Thai presentation-form glyphs in the Private Use Area (the Microsoft/Apple Thai
# variant range, seen in BBL text layers) -> standard Thai codepoints.
# Verified against the cmap of the TLWG Garuda font (fonts-tlwg, installed in the
# devcontainer): U+F700 -> uni0E10.descless, U+F705..F709 -> uni0E48..0E4C.low_left,
# U+F70A..F70E -> uni0E48..0E4C.low, U+F713..F717 -> uni0E48..0E4C.left,
# U+F718..F71A -> uni0E38..0E3A.low, and so on; each glyph name's base is the target here.
_TONE_MARKS = [0x0E48, 0x0E49, 0x0E4A, 0x0E4B, 0x0E4C]
_PUA_MAP: dict[int, int] = {
    0xF700: 0x0E10,
    0xF701: 0x0E34,
    0xF702: 0x0E35,
    0xF703: 0x0E36,
    0xF704: 0x0E37,
    **{0xF705 + i: cp for i, cp in enumerate(_TONE_MARKS)},
    **{0xF70A + i: cp for i, cp in enumerate(_TONE_MARKS)},
    0xF70F: 0x0E0D,
    0xF710: 0x0E31,
    0xF711: 0x0E4D,
    0xF712: 0x0E47,
    **{0xF713 + i: cp for i, cp in enumerate(_TONE_MARKS)},
    0xF718: 0x0E38,
    0xF719: 0x0E39,
    0xF71A: 0x0E3A,
}
_THAI_DIGITS = {0x0E50 + i: ord(str(i)) for i in range(10)}
_ZERO_WIDTH = {cp: None for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)}
_TRANSLATE = {**_PUA_MAP, **_THAI_DIGITS, **_ZERO_WIDTH}

_WS_RE = re.compile(r"\s+")


def canonical_text(s: str) -> str:
    """NFC, PUA Thai glyphs -> standard, Thai digits -> 0-9, zero-width removed,
    whitespace runs collapsed to one space, stripped."""
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_TRANSLATE)
    # PUA mapping can put a mark next to a base that NFC would compose differently.
    s = unicodedata.normalize("NFC", s)
    return _WS_RE.sub(" ", s).strip()


_FIGURE_RE = re.compile(r"<figure\b[^>]*>.*?</figure\s*>", re.IGNORECASE | re.DOTALL)
_PAGE_NUMBER_TAG_RE = re.compile(r"</?page_number\s*>", re.IGNORECASE)
_BREAK_TAG_RE = re.compile(r"<br\s*/?>|</p\s*>|</tr\s*>|</t[dh]\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^[ \t]*#+[ \t]*", re.MULTILINE)
_RULE_LINE_RE = re.compile(r"^[ \t|:\-]*-[ \t|:\-]*$", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"\*+|(?<!\w)_+|_+(?!\w)")


def plain_text(s: str) -> str:
    """The markup-free view used for CER/WER on both sides.

    Drops figures (with content), keeps page numbers, turns cell/row/paragraph/line
    breaks into spaces, strips all other tags and unescapes entities, removes markdown
    markup (headings, emphasis, pipes, table rule lines, `$$`), then `canonical_text`.
    """
    s = _FIGURE_RE.sub(" ", s)
    s = _PAGE_NUMBER_TAG_RE.sub(" ", s)
    s = _BREAK_TAG_RE.sub(" ", s)
    s = _TAG_RE.sub("", s)
    s = html.unescape(s)
    s = _RULE_LINE_RE.sub("", s)
    s = _HEADING_RE.sub("", s)
    s = s.replace("$$", " ").replace("|", " ")
    s = _EMPHASIS_RE.sub("", s)
    return canonical_text(s)
