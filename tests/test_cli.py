"""
Tests for AuditorAI CLI commands.
"""

from unittest.mock import patch
import pytest
from auditorai.cli.main import main


def test_cli_run_and_validate(tmp_path):
    model_dir = str(tmp_path / "models")
    
    # Test CLI run
    with patch("sys.argv", ["auditorai", "run", "--data", "breast_cancer", "--save-dir", model_dir]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0

    # Test CLI validate
    with patch("sys.argv", ["auditorai", "validate", "--adapter-path", model_dir, "--data", "breast_cancer"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
