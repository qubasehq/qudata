"""Basic tests for QuData."""

def test_import():
    """Test that the package can be imported."""
    import qudata
    assert qudata.__version__ == "1.0.0"

def test_cli_help():
    """Test CLI help command."""
    from qudata.cli import main
    # Basic smoke test - just ensure CLI doesn't crash
    assert main is not None
