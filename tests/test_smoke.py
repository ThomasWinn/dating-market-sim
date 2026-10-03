import rankers
import sim


def test_packages_import():
    assert sim.__name__ == "sim"
    assert rankers.__name__ == "rankers"
