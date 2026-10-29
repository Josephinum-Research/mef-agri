from PyQt5.QtWidgets import QMenu


class _TEXT:
    MENU_ZDEL = 'DELETE'
    MENU_ZRENDER = 'RENDER'


class ZoningMenu(QMenu):
    def __init__(self):
        super().__init__()
        zdel = self.addAction(_TEXT.MENU_ZDEL)
        zdel.triggered.connect(self._ondel)
        rndr = self.addAction(_TEXT.MENU_ZRENDER)
        rndr.triggered.connect(self._onrndr)

        self._zsel:str = None
        self._f_ondel:function = None
        self._f_onrndr:function = None

    @property
    def selected_zoning(self) -> str:
        """
        settable

        :return: zoning-identifier from folder structure which has been right-clicked by the user
        :rtype: str
        """
        return self._zsel

    @selected_zoning.setter
    def selected_zoning(self, zoning):
        self._zsel = zoning

    @property
    def on_delete(self) -> function:
        """
        settable

        :return: ZoningTab-method which is called when user clicks on the delete-entry of the context menu (has to accept a string-parameter beside ``self`` representing the selected zoning-identifier from the project folder structure)
        :rtype: function
        """
        return self._f_ondel

    @on_delete.setter
    def on_delete(self, func):
        self._f_ondel = func

    @property
    def on_render(self) -> function:
        """
        settable

        :return: ZoningTab-method which is called when user clicks on the render-entry of the context menu (has to accept a string-parameter beside ``self`` representing the selected zoning-identifier from the project folder structure)
        :rtype: function
        """
        return self._f_onrndr

    @on_render.setter
    def on_render(self, func):
        self._f_onrndr = func

    def _ondel(self):
        if self._check_zoning():
            self.on_delete(self._zsel)

    def _onrndr(self):
        if self._check_zoning():
            self.on_render(self._zsel)

    def _check_zoning(self) -> bool:
        if self._zsel is None:
            return False
        elif not isinstance(self._zsel, str):
            return False
        else:
            return True
