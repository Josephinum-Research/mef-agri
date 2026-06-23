from PyQt5 import QtWidgets, QtGui

from ..utils.store import AppStore


class CustomTabWidget(QtWidgets.QWidget):
    def __init__(self, parent, store):
        super().__init__(parent)
        self._init:bool = False
        self._store:AppStore = store

    @property
    def initialized(self) -> bool:
        """
        :return: flag if ``init_tab`` method has been called
        :rtype: bool
        """
        return self._init
    
    @property
    def store(self) -> AppStore:
        """
        :return: app-store which contains app-wide-required stuff
        :rtype: AppStore
        """
        return self._store
    
    def init_tab(self):
        self._init = True

    def tab_clicked(self):
        pass



class ComboBox(QtWidgets.QComboBox):
    """
    Custom combo box class which enables setting a non-selectable placeholder 
    text.
    This class can be used exactly like ``PyQt5.QtWidgets.QComboBox``.
    
    See the following links for more explanation:

    * https://stackoverflow.com/questions/65826378/how-do-i-use-qcombobox-setplaceholdertext/65830989#65830989
    * https://code.qt.io/cgit/qt/qtbase.git/tree/src/widgets/widgets/qcombobox.cpp?h=5.15.2#n3173
    
    """
    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))

        # draw the combobox frame, focusrect and selected etc.
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)

        if self.currentIndex() < 0:
            opt.palette.setBrush(
                QtGui.QPalette.ButtonText,
                opt.palette.brush(QtGui.QPalette.ButtonText).color().lighter(),
            )
            if self.placeholderText():
                opt.currentText = self.placeholderText()

        # draw the icon and text
        painter.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt)