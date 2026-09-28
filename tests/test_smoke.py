"""Pytest wrapper for end-to-end smoke test."""

import pytest
from scripts.smoke_test import run_smoke_test

def test_end_to_end_smoke():
    success = run_smoke_test()
    assert success is True
