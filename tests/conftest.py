import pytest


@pytest.fixture
def no_blas(monkeypatch):
    """Anvil's own kernels for every matrix product (tests of the NEON schedules)."""
    monkeypatch.setenv("ANVIL_BLAS", "0")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a test that takes more than a few seconds")
