# learn PYOpenCL (C) 2020 Serhii Lysovenko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# If matplotlib contributes to a project that leads to a scientific
# publication, please acknowledge this fact by citing the project.
# You can use this BibTeX entry:
# @Article{Hunter:2007,
#   Author    = {Hunter, J. D.},
#   Title     = {Matplotlib: A 2D graphics environment},
#   Journal   = {Computing In Science \& Engineering},
#   Volume    = {9},
#   Number    = {3},
#   Pages     = {90--95},
#   abstract  = {Matplotlib is a 2D graphics package used for Python
#   for application development, interactive scripting, and
#   publication-quality image generation across user
#   interfaces and operating systems.},
#   publisher = {IEEE COMPUTER SOC},
#   year      = 2007
# }
"""Display plots"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QSizePolicy, QAction, QMainWindow)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from misc import lena


class Canvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = fig = Figure(figsize=(width, height), dpi=dpi)
        self.mk_axes()
        super().__init__(fig)
        self.setParent(parent)

        super().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        super().updateGeometry()

    def mk_axes(self):
        self.axes1 = self.figure.add_subplot(111)
        self.axes1.grid(True)
        self.axes2 = None
        self.axes1.set_xlabel(r'$s,\, \AA^{-1}$')
        self.axes1.set_ylabel('Intensity')

    def draw(self, dset=None):
        if dset is None:
            return super().draw()
        self.figure.clear()
        self.axes1 = self.figure.add_subplot(111)
        if "img" in dset:
            if len(dset["img"].shape) < 3:
                cmap = 'gray'
            else:
                cmap = None
            self.axes1.imshow(dset["img"], cmap=cmap)
        super().draw()


class PlotWindow(QMainWindow):
    """Plot and toolbar"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot some examples")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.canvas = Canvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(self.toolbar)
        self.setCentralWidget(self.canvas)
        self.mbar = self.menuBar()
        self.interactor = None
        examples = self.mbar.addMenu("Examples")
        examples.addAction(QAction("Rotate", self, triggered=self._set_rotate))
        examples.addAction(QAction("Mandelbrot", self, triggered=print))
        self.draw({"img": lena()})

    def closeEvent(self, event):
        """finalize"""

    def _set_rotate(self, *args):
        from rotation import Interactor
        self.interactor = Interactor()

    def draw(self, plt):
        self.canvas.draw(plt)

    def interact(self, action):
        if self.interactor is None:
            return print(action)
        self.draw(self.interactor(action))

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up:
            self.interact("Key_UP")
        elif key == Qt.Key_Down:
            self.interact("Key_Down")
        elif key == Qt.Key_Left:
            self.interact("Key_Left")
        elif key == Qt.Key_Right:
            self.interact("Key_Right")
        elif key == Qt.Key_PageUp:
            self.interact("Key_PageUp")
        elif key == Qt.Key_PageDown:
            self.interact("Key_PageDown")
        elif key == Qt.Key_Insert:
            self.interact("Key_Insert")
        elif key == Qt.Key_Space or key == Qt.Key_Enter:
            pass
        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("XRCEA")
    plt = PlotWindow()
    plt.show()
    sys.exit(app.exec_())
