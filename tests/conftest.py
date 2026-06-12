"""Shared test plumbing.

Tests load YOUR files from modules/ via r2r.course.load. Until you
implement a TODO, its tests fail with your NotImplementedError — that's
the red light turning green as you build, module by module.

Run one module's tests at a time:

    pytest tests/test_module03.py -q
"""

import pytest

from r2r.course import load


@pytest.fixture
def course():
    return load
