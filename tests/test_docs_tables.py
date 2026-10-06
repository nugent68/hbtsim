"""The parameter tables in README.md and docs/ are rendered from the
catalog (scripts/catalog_tables.py); this fails when they drift."""

import importlib.util
import pathlib


def test_catalog_tables_are_current(capsys):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("catalog_tables", root / "scripts" / "catalog_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([]) == 0, capsys.readouterr().out
