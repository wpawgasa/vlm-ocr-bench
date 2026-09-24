"""Document classes and the class registry (docparse task 2.4).

A class file (`configs/docparse/classes/<id>.yaml`) describes one document class: its
fields, their value types, label aliases, region hints, validators and `required_when`
conditions. The `bank_statement` class's aliases are resolved from the `ocr_bench`
statement mapper's own tables at load time (`aliases_from`), so the two never drift
apart. The registry is the shared source of truth for classification, verification and
extraction (document-registry spec).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ocr_bench.config import BankOverrides
from ocr_bench.metrics.arithmetic import check_file
from ocr_bench.normalize import statement as stmt
from ocr_bench.schemas import StatementFile

FieldType = Literal["text", "amount", "date", "account", "table"]


class RegistryError(ValueError):
    """An invalid class file or directory; the message starts with the file path."""


# --- raw YAML shapes -------------------------------------------------------------------------


class RegionHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: Literal["first", "last", "any"] = "any"
    box: tuple[float, float, float, float]  # x0, y0, x1, y1, fractions of the page

    @model_validator(mode="after")
    def _check_box(self) -> "RegionHint":
        x0, y0, x1, y1 = self.box
        for v in (x0, y0, x1, y1):
            if not 0.0 <= v <= 1.0:
                raise ValueError("region_hint.box values must be in [0, 1]")
        if x0 >= x1:
            raise ValueError("region_hint.box: x0 must be less than x1")
        if y0 >= y1:
            raise ValueError("region_hint.box: y0 must be less than y1")
        return self


class BankIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bank_in: list[str]


class BankNotIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bank_not_in: list[str]


RequiredWhen = Literal["always"] | BankIn | BankNotIn


class FieldSpec(BaseModel):
    """The raw YAML shape of one field."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: FieldType
    aliases: list[str] = []
    aliases_from: str | None = None
    columns: dict[str, str] | None = None
    region_hint: RegionHint | None = None
    validators: list[str] = []
    required_when: RequiredWhen = "always"

    @model_validator(mode="after")
    def _check_shape(self) -> "FieldSpec":
        if self.type == "table":
            if not self.columns:
                raise ValueError("a 'table' field requires non-empty `columns`")
            if self.aliases_from is not None:
                raise ValueError("`aliases_from` is not allowed for a 'table' field")
        else:
            if self.columns is not None:
                raise ValueError(f"`columns` is not allowed for a '{self.type}' field")
        return self


class ClassSpec(BaseModel):
    """The raw YAML shape of a class file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    banks_from: str | None = None
    cross_validators: list[str] = []
    fields: list[FieldSpec]


# --- resolved definitions ---------------------------------------------------------------------


@dataclass(frozen=True)
class FieldDef:
    name: str
    type: str
    aliases: tuple[str, ...]
    region_hint: RegionHint | None
    validators: tuple[str, ...]
    required_when: RequiredWhen
    columns: dict[str, tuple[str, ...]] | None

    def is_required(self, bank: str | None) -> bool:
        rw = self.required_when
        if rw == "always":
            return True
        known = bank is not None and bank != "unknown"
        if isinstance(rw, BankIn):
            return known and bank in rw.bank_in
        if isinstance(rw, BankNotIn):
            return not known or bank not in rw.bank_not_in
        return True  # pragma: no cover — RequiredWhen has no other case


@dataclass(frozen=True)
class DocumentClass:
    id: str
    description: str
    banks: tuple[str, ...]
    bank_overrides: dict[str, BankOverrides]
    fields: tuple[FieldDef, ...]
    cross_validators: tuple[str, ...]
    _fields_by_name: dict[str, FieldDef] = field(
        default_factory=dict, repr=False, compare=False, init=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_fields_by_name", {f.name: f for f in self.fields})

    def field(self, name: str) -> FieldDef:
        return self._fields_by_name[name]

    def required_fields(self, bank: str | None) -> list[FieldDef]:
        return [f for f in self.fields if f.is_required(bank)]

    def validate_value(self, field_name: str, value: str) -> bool:
        f = self.field(field_name)
        return all(FIELD_VALIDATORS[name](value, self) for name in f.validators)

    def validate_statement(self, statement: StatementFile) -> bool:
        return all(CROSS_VALIDATORS[name](statement) for name in self.cross_validators)


# --- validators --------------------------------------------------------------------------------

_ACCOUNT_NO_RE = re.compile(r"[0-9xX*\- ]+")


def _validate_date(value: str, _doc_class: DocumentClass) -> bool:
    return stmt.parse_date(value) is not None


def _validate_amount(value: str, _doc_class: DocumentClass) -> bool:
    return stmt.parse_decimal(value) is not None


def _validate_account_number(value: str, _doc_class: DocumentClass) -> bool:
    v = value.strip()
    if not _ACCOUNT_NO_RE.fullmatch(v):
        return False
    return sum(ch.isdigit() for ch in v) >= 4


def _validate_non_empty(value: str, _doc_class: DocumentClass) -> bool:
    return value.strip() != ""


def _validate_known_bank(value: str, doc_class: DocumentClass) -> bool:
    return value in doc_class.banks


FIELD_VALIDATORS: dict[str, Callable[[str, DocumentClass], bool]] = {
    "date": _validate_date,
    "amount": _validate_amount,
    "account_number": _validate_account_number,
    "non_empty": _validate_non_empty,
    "known_bank": _validate_known_bank,
}


def _validate_running_balance(statement: StatementFile) -> bool:
    return check_file(statement).file_reconciles


CROSS_VALIDATORS: dict[str, Callable[[StatementFile], bool]] = {
    "running_balance": _validate_running_balance,
}


# --- registry ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Registry:
    classes: dict[str, DocumentClass]

    def get(self, class_id: str) -> DocumentClass:
        return self.classes[class_id]

    def ids(self) -> list[str]:
        return sorted(self.classes)

    def descriptions(self) -> dict[str, str]:
        return {cid: c.description for cid, c in self.classes.items()}


def _resolve_aliases(path: Path, name: str, spec: FieldSpec) -> tuple[str, ...]:
    aliases: list[str] = list(spec.aliases)
    if spec.aliases_from is not None:
        prefix, _, mapper_field = spec.aliases_from.partition(":")
        if prefix != "header" or not mapper_field:
            raise RegistryError(
                f"{path}: field {name!r}: aliases_from must be 'header:<mapper_field>', "
                f"got {spec.aliases_from!r}"
            )
        labels = [label for field_name, label in stmt.HEADER_LABELS if field_name == mapper_field]
        if not labels:
            raise RegistryError(
                f"{path}: field {name!r}: aliases_from references unknown header field "
                f"{mapper_field!r}"
            )
        aliases.extend(labels)
    seen: list[str] = []
    for a in aliases:
        if a not in seen:
            seen.append(a)
    return tuple(seen)


def _resolve_columns(
    path: Path, name: str, columns: dict[str, str] | None
) -> dict[str, tuple[str, ...]] | None:
    if columns is None:
        return None
    resolved: dict[str, tuple[str, ...]] = {}
    for col_name, ref in columns.items():
        prefix, _, role = ref.partition(":")
        if prefix != "column" or not role:
            raise RegistryError(
                f"{path}: field {name!r}: column {col_name!r} must reference "
                f"'column:<role>', got {ref!r}"
            )
        keywords = stmt.COLUMN_KEYWORDS.get(role)
        if keywords is None:
            raise RegistryError(
                f"{path}: field {name!r}: column {col_name!r} references unknown role {role!r}"
            )
        resolved[col_name] = tuple(keywords)
    return resolved


def _resolve_field(path: Path, spec: FieldSpec, banks: tuple[str, ...]) -> FieldDef:
    name = spec.name
    for v in spec.validators:
        if v not in FIELD_VALIDATORS:
            raise RegistryError(f"{path}: field {name!r}: unknown validator {v!r}")
    rw = spec.required_when
    if isinstance(rw, BankIn):
        unknown = [b for b in rw.bank_in if b not in banks]
    elif isinstance(rw, BankNotIn):
        unknown = [b for b in rw.bank_not_in if b not in banks]
    else:
        unknown = []
    if unknown:
        raise RegistryError(
            f"{path}: field {name!r}: required_when names unknown bank(s) {unknown}"
        )
    return FieldDef(
        name=name,
        type=spec.type,
        aliases=_resolve_aliases(path, name, spec),
        region_hint=spec.region_hint,
        validators=tuple(spec.validators),
        required_when=rw,
        columns=_resolve_columns(path, name, spec.columns),
    )


def _load_banks(
    path: Path, banks_from: str | None
) -> tuple[tuple[str, ...], dict[str, BankOverrides]]:
    if banks_from is None:
        return (), {}
    banks_path = (path.parent / banks_from).resolve()
    if not banks_path.exists():
        raise RegistryError(f"{path}: banks_from {banks_from!r} does not exist ({banks_path})")
    try:
        with open(banks_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path}: banks_from {banks_from!r}: {exc}") from exc
    banks = tuple(data.get("banks") or [])
    raw_overrides = data.get("overrides") or {}
    overrides: dict[str, BankOverrides] = {}
    for bank, raw in raw_overrides.items():
        try:
            overrides[bank] = BankOverrides.model_validate(raw or {})
        except ValidationError as exc:
            raise RegistryError(f"{path}: banks_from {banks_from!r}: bank {bank!r}: {exc}") from exc
    return banks, overrides


def load_class(path: Path) -> DocumentClass:
    """Load and validate one class file; raises `RegistryError` naming `path`."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RegistryError(f"{path}: expected a mapping at the top level")
    raw_fields = data.get("fields")
    if not isinstance(raw_fields, list):
        raise RegistryError(f"{path}: `fields` must be a list of field definitions")
    top_level = {k: v for k, v in data.items() if k != "fields"}

    # Validate top-level keys (everything but `fields`) together with an empty field list
    # first, so a top-level error is reported without a field-by-field walk; then validate
    # each field entry separately so a bad one names itself.
    try:
        top_spec = ClassSpec.model_validate({**top_level, "fields": []})
    except ValidationError as exc:
        raise RegistryError(f"{path}: {exc}") from exc

    field_specs: list[FieldSpec] = []
    for i, raw in enumerate(raw_fields or []):
        raw_name = raw.get("name") if isinstance(raw, dict) else None
        label = raw_name if raw_name else f"#{i}"
        try:
            field_specs.append(FieldSpec.model_validate(raw))
        except ValidationError as exc:
            raise RegistryError(f"{path}: field {label!r}: {exc}") from exc

    spec = top_spec.model_copy(update={"fields": field_specs})

    if spec.id != path.stem:
        raise RegistryError(f"{path}: id {spec.id!r} does not match file name {path.stem!r}")

    names = [f.name for f in spec.fields]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise RegistryError(f"{path}: duplicate field name {name!r}")
        seen.add(name)

    for cv in spec.cross_validators:
        if cv not in CROSS_VALIDATORS:
            raise RegistryError(f"{path}: cross_validators: unknown validator {cv!r}")

    banks, overrides = _load_banks(path, spec.banks_from)
    fields = tuple(_resolve_field(path, fs, banks) for fs in spec.fields)

    return DocumentClass(
        id=spec.id,
        description=spec.description,
        banks=banks,
        bank_overrides=overrides,
        fields=fields,
        cross_validators=tuple(spec.cross_validators),
    )


def load_registry(classes_dir: Path) -> Registry:
    """Load every `*.yaml` class file in `classes_dir`, sorted by file name."""
    if not classes_dir.is_dir():
        raise RegistryError(f"{classes_dir}: no such directory")
    classes: dict[str, DocumentClass] = {}
    sources: dict[str, Path] = {}
    for path in sorted(classes_dir.glob("*.yaml")):
        doc_class = load_class(path)
        if doc_class.id in classes:
            raise RegistryError(
                f"{path}: duplicate class id {doc_class.id!r} (also defined in "
                f"{sources[doc_class.id]})"
            )
        classes[doc_class.id] = doc_class
        sources[doc_class.id] = path
    return Registry(classes=classes)
