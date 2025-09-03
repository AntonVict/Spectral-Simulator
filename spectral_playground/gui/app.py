"""Simple wrapper to launch the main GUI application."""

try:
    # Try relative import first (when used as module)
    from .main_gui import main
except ImportError:
    # Fall back to absolute import (when run directly)
    from spectral_playground.gui.main_gui import main

if __name__ == "__main__":
    main()
