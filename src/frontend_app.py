"""Compatibility entry point for the SnapSign dashboard."""

from frontend import SnapSignDashboard
from PyQt5.QtWidgets import QApplication
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnapSignDashboard()
    window.show()
    sys.exit(app.exec_())
