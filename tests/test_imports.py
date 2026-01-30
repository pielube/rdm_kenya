"""Smoke tests to ensure core modules import successfully."""

import rdm_kenya
from rdm_kenya import analysis, experiment, postprocessing, utils, workflow_utils


def test_core_module_imports():
    assert rdm_kenya.__all__ is not None
    assert analysis is not None
    assert experiment is not None
    assert postprocessing is not None
    assert utils is not None
    assert workflow_utils is not None
