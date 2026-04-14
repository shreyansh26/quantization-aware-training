import importlib


def test_package_imports() -> None:
    module = importlib.import_module("qat")
    assert module is not None
