from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QVBoxLayout, QLabel, QStylePainter, 
    QStyleOptionComboBox, QStyle, QLayout
)
from PyQt5.QtCore import Qt

from ..utils.store import AppStore


class _TEXT:
    LBL_INIT = 'no project selected!'


class BaseTab(QWidget):
    def __init__(self, parent, store):
        super().__init__(parent)
        self._store:AppStore = store

    @property
    def store(self) -> AppStore:
        """
        :return: app-store which contains app-wide-required stuff
        :rtype: AppStore
        """
        return self._store


class NonProjectTab(BaseTab):
    def __init__(self, parent, store):
        super().__init__(parent, store)
        # internal variables
        self._init:bool = False
        self._l_main:QLayout = None

        # initial layout/appearance
        self._l_init = QVBoxLayout()
        self._lbl_init = QLabel(_TEXT.LBL_INIT)
        self._lbl_init.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._l_init.addWidget(self._lbl_init)

        self.setLayout(self._l_init)

    @property
    def initialized(self) -> bool:
        """
        :return: flag if ``init_tab`` method has been called
        :rtype: bool
        """
        return self._init
    
    @property
    def layout_main(self) -> QLayout:
        """
        Settable

        :return: main layout of tab which is visible after project has been selected
        :rtype: QLayout
        """
        return self._l_main

    @layout_main.setter
    def layout_main(self, layout):
        self._l_main = layout
    
    def init_tab(self):
        self._init = True
        self._lbl_init.setVisible(False)
        if self._l_main is not None:
            self._l_init.addLayout(self._l_main)



class ComboBox(QComboBox):
    """
    Custom combo box class which enables setting a non-selectable placeholder 
    text.
    This class can be used exactly like ``PyQt5.QtWidgets.QComboBox``.
    
    See the following links for more explanation:

    * https://stackoverflow.com/questions/65826378/how-do-i-use-qcombobox-setplaceholdertext/65830989#65830989
    * https://code.qt.io/cgit/qt/qtbase.git/tree/src/widgets/widgets/qcombobox.cpp?h=5.15.2#n3173
    
    """
    def __init__(self, placeholder_text:str=None, parent=None):
        super().__init__(parent)
        if placeholder_text is not None:
            self.setPlaceholderText(placeholder_text)
            self.setCurrentIndex(-1)

    def paintEvent(self, event):
        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))

        # draw the combobox frame, focusrect and selected etc.
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)

        if self.currentIndex() < 0:
            opt.palette.setBrush(
                QtGui.QPalette.ButtonText,
                opt.palette.brush(QtGui.QPalette.ButtonText).color().lighter(),
            )
            if self.placeholderText():
                opt.currentText = self.placeholderText()

        # draw the icon and text
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)