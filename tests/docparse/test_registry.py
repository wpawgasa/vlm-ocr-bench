"""Tests for the document class registry (docparse task 2.4, document-registry spec).

The repo is public: no real bank-statement content appears here, only made-up values.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from docparse.registry import RegistryError, load_class, load_registry
from ocr_bench.normalize import statement as stmt
from ocr_bench.schemas import StatementFile, StatementPage, StatementRow

SHIPPED_CLASSES = Path(__file__).resolve().parents[2] / "configs" / "docparse" / "classes"

BANKS_YAML = """\
banks: [kbank, scb]
overrides:
  kbank:
    amount_column_split: true
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --- scenario 1: valid registry loads -----------------------------------------------------


class TestValidRegistryLoads:
    def test_bank_statement_exposed(self):
        registry = load_registry(SHIPPED_CLASSES)
        assert "bank_statement" in registry.ids()

    def test_account_no_field(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        account_no = bc.field("account_no")
        assert "account no" in account_no.aliases
        assert "เลขที่บัญชี" in account_no.aliases
        assert account_no.validators == ("account_number",)
        assert account_no.region_hint is not None
        assert account_no.region_hint.page == "first"

    def test_transactions_table(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        transactions = bc.field("transactions")
        assert transactions.type == "table"
        assert set(transactions.columns) == {"date", "description", "debit", "credit", "balance"}
        assert "withdrawal" in transactions.columns["debit"]

    def test_banks_and_overrides(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        assert "kbank" in bc.banks
        assert "ttb" in bc.banks
        assert bc.bank_overrides["ttb"].signed_amount is True

    def test_descriptions_non_empty(self):
        registry = load_registry(SHIPPED_CLASSES)
        assert registry.descriptions()["bank_statement"]


# --- scenario 2: invalid field type rejected ----------------------------------------------


class TestInvalidFieldTypeRejected:
    def test_unknown_type(self, tmp_path):
        _write(
            tmp_path / "bad.yaml",
            """\
id: bad
description: a bad class
fields:
  - name: currency
    type: currency_code
""",
        )
        with pytest.raises(RegistryError) as exc_info:
            load_class(tmp_path / "bad.yaml")
        message = str(exc_info.value)
        assert str(tmp_path / "bad.yaml") in message
        assert "currency" in message


# --- scenario 3: new class without code changes -------------------------------------------


class TestNewClassWithoutCodeChanges:
    def test_invoice_class_loads(self, tmp_path):
        _write(
            tmp_path / "invoice.yaml",
            """\
id: invoice
description: An invoice document.
fields:
  - name: vendor
    type: text
    aliases: [vendor, "ผู้ขาย"]
    validators: [non_empty]
""",
        )
        registry = load_registry(tmp_path)
        assert registry.ids() == ["invoice"]
        assert "invoice" in registry.descriptions()


# --- scenario 4: KBank opening balance ------------------------------------------------------


class TestKBankOpeningBalance:
    def test_is_required(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        opening = bc.field("opening_balance")
        assert opening.is_required("kbank") is False
        assert opening.is_required("scb") is True
        assert opening.is_required(None) is True

    def test_required_fields(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        kbank_required = {f.name for f in bc.required_fields("kbank")}
        scb_required = {f.name for f in bc.required_fields("scb")}
        assert "opening_balance" not in kbank_required
        assert "opening_balance" in scb_required


# --- scenario: aliases shared with the benchmark mapper -------------------------------------


class TestSharedAlias:
    def test_extended_header_labels_picked_up(self, monkeypatch):
        monkeypatch.setattr(
            stmt,
            "HEADER_LABELS",
            stmt.HEADER_LABELS + (("account_no", "หมายเลขบัญชีทดสอบ"),),
        )
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        assert "หมายเลขบัญชีทดสอบ" in bc.field("account_no").aliases


# --- validators -------------------------------------------------------------------------------


class TestFieldValidators:
    @pytest.fixture
    def bank_statement(self):
        return load_registry(SHIPPED_CLASSES).get("bank_statement")

    def test_date(self, bank_statement):
        assert bank_statement.validate_value("period_start", "01/10/2022") is True
        assert bank_statement.validate_value("period_start", "not a date") is False

    def test_amount(self, bank_statement):
        assert bank_statement.validate_value("closing_balance", "1,234.50") is True
        assert bank_statement.validate_value("closing_balance", "abc") is False

    def test_account_number(self, bank_statement):
        assert bank_statement.validate_value("account_no", "123-4-56789-0") is True
        assert bank_statement.validate_value("account_no", "xxx-x-x1234-x") is True
        assert bank_statement.validate_value("account_no", "12") is False
        assert bank_statement.validate_value("account_no", "abc1234") is False

    def test_bank(self, bank_statement):
        assert bank_statement.validate_value("bank", "kbank") is True
        assert bank_statement.validate_value("bank", "citibank") is False


# --- validate_statement -----------------------------------------------------------------------


def _row(date, *, debit=None, credit=None, balance):
    return StatementRow(
        date=date,
        debit=None if debit is None else Decimal(debit),
        credit=None if credit is None else Decimal(credit),
        balance=Decimal(balance),
        page_no=1,
    )


class TestValidateStatement:
    def test_reconciling_statement(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        page = StatementPage(
            page_no=1,
            opening_balance=Decimal("1000.00"),
            rows=[
                _row("2024-01-01", debit="200.00", balance="800.00"),
                _row("2024-01-02", credit="50.00", balance="850.00"),
            ],
        )
        sf = StatementFile(file_id="t", bank="scb", pages=[page])
        assert bc.validate_statement(sf) is True

    def test_non_reconciling_statement(self):
        bc = load_registry(SHIPPED_CLASSES).get("bank_statement")
        page = StatementPage(
            page_no=1,
            opening_balance=Decimal("1000.00"),
            rows=[
                _row("2024-01-01", debit="200.00", balance="800.00"),
                _row("2024-01-02", credit="50.00", balance="999.00"),
            ],
        )
        sf = StatementFile(file_id="t", bank="scb", pages=[page])
        assert bc.validate_statement(sf) is False


# --- scenario 8: loading errors --------------------------------------------------------------


class TestLoadingErrors:
    def test_id_mismatch(self, tmp_path):
        path = _write(
            tmp_path / "invoice.yaml",
            "id: not_invoice\ndescription: x\nfields: []\n",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_missing_fields_key(self, tmp_path):
        path = _write(tmp_path / "invoice.yaml", "id: invoice\ndescription: x\n")
        with pytest.raises(RegistryError, match="fields"):
            load_class(path)

    def test_duplicate_field_names(self, tmp_path):
        path = _write(
            tmp_path / "dup.yaml",
            """\
id: dup
description: x
fields:
  - name: vendor
    type: text
    validators: [non_empty]
  - name: vendor
    type: text
    validators: [non_empty]
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_unknown_field_validator(self, tmp_path):
        path = _write(
            tmp_path / "badval.yaml",
            """\
id: badval
description: x
fields:
  - name: vendor
    type: text
    validators: [not_a_real_validator]
""",
        )
        with pytest.raises(RegistryError) as exc_info:
            load_class(path)
        message = str(exc_info.value)
        assert str(path) in message
        assert "vendor" in message

    def test_unknown_cross_validator(self, tmp_path):
        path = _write(
            tmp_path / "badcross.yaml",
            """\
id: badcross
description: x
cross_validators: [not_a_real_validator]
fields:
  - name: vendor
    type: text
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_unknown_aliases_from_field(self, tmp_path):
        path = _write(
            tmp_path / "badalias.yaml",
            """\
id: badalias
description: x
fields:
  - name: vendor
    type: text
    aliases_from: header:no_such_field
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_unknown_column_role(self, tmp_path):
        path = _write(
            tmp_path / "badcolumn.yaml",
            """\
id: badcolumn
description: x
fields:
  - name: items
    type: table
    columns:
      qty: column:no_such_role
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_table_without_columns(self, tmp_path):
        path = _write(
            tmp_path / "notable.yaml",
            """\
id: notable
description: x
fields:
  - name: items
    type: table
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_text_field_with_columns(self, tmp_path):
        path = _write(
            tmp_path / "badtext.yaml",
            """\
id: badtext
description: x
fields:
  - name: vendor
    type: text
    columns:
      a: column:date
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_required_when_unknown_bank(self, tmp_path):
        _write(tmp_path / "banks.yaml", BANKS_YAML)
        path = _write(
            tmp_path / "withbanks.yaml",
            """\
id: withbanks
description: x
banks_from: banks.yaml
fields:
  - name: vendor
    type: text
    required_when: {bank_in: [citibank]}
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_banks_from_missing_file(self, tmp_path):
        path = _write(
            tmp_path / "missingbanks.yaml",
            """\
id: missingbanks
description: x
banks_from: no_such_file.yaml
fields: []
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)

    def test_region_hint_bad_box(self, tmp_path):
        path = _write(
            tmp_path / "badbox.yaml",
            """\
id: badbox
description: x
fields:
  - name: vendor
    type: text
    region_hint: {page: first, box: [0.5, 0.0, 0.2, 1.0]}
""",
        )
        with pytest.raises(RegistryError, match=str(path)):
            load_class(path)


# --- scenario 9: load_registry on directories ------------------------------------------------


class TestLoadRegistryDirectories:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(RegistryError):
            load_registry(tmp_path / "no_such_dir")

    def test_multiple_classes_load(self, tmp_path):
        _write(tmp_path / "a.yaml", "id: a\ndescription: x\nfields: []\n")
        _write(tmp_path / "b.yaml", "id: b\ndescription: y\nfields: []\n")
        registry = load_registry(tmp_path)
        assert registry.ids() == ["a", "b"]

    def test_empty_dir_gives_empty_registry(self, tmp_path):
        registry = load_registry(tmp_path)
        assert registry.ids() == []
