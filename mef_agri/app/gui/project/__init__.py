import types

from ....data.project import ProjectData


class ProjectDataGUI(ProjectData):
    ADD_DATA_UI_PROPS = [
        'processed_field', 'processed_interface', 'progress', 'add_data_error', 
        'add_data_success'
    ]
    
    def __init__(self, project_dir, gpkg_name):
        super().__init__(project_dir, gpkg_name)
        self._gui_funcs:dict = {}
        for prop in self.ADD_DATA_UI_PROPS:
            self._gui_funcs[prop] = []

    @ProjectData.processed_field.setter
    def processed_field(self, fname):
        self._pfld = fname
        self._loop_funcs(self._gui_funcs['processed_field'], fname)

    @ProjectData.processed_interface.setter
    def processed_interface(self, iname):
        self._pintf = iname
        self._loop_funcs(self._gui_funcs['processed_interface'], iname)

    @ProjectData.progress.setter
    def progress(self, pstate):
        self._prgr = pstate
        self._loop_funcs(self._gui_funcs['progress'], pstate)

    @ProjectData.add_data_error.setter
    def add_data_error(self, err):
        self._aderr = err
        self._loop_funcs(self._gui_funcs['add_data_error'], err)

    @ProjectData.add_data_success.setter
    def add_data_success(self, succ):
        self._adsucc = succ
        self._loop_funcs(self._gui_funcs['add_data_success'], succ)

    def register_add_data_interaction(
            self, prop:property | str, func:function
        ):
        """
        Register handlers if one of the following properties is changed

        * :func:`processed_field`
        * :func:`processed_interface`
        * :func:`progress`
        * :func:`add_data_error`
        * :func:`add_data_success`

        ``prop`` can be provided as string (e.g. ``'progress'``) or as 
        property (e.g. ``ProjectData.progress``).
        ``func`` has to accept one argument, being the newly set value of 
        ``prop``.

        :param prop: specify at which property-change ``func`` will be called
        :type prop: property | str
        :param func: function which should be called when ``prop`` changes
        :type func: function
        """
        prop = self._check_prop(prop)
        self._gui_funcs[prop].append(func)

    def remove_add_data_interaction(
            self, prop:property | str, func:function | str
        ):
        prop = self._check_prop(prop)
        if isinstance(func, types.FunctionType):
            func = func.__name__
        for ix in range(len(self._gui_funcs[prop])):
            fi = self._gui_funcs[prop][ix]
            if fi.__name__ == func:
                del self._gui_funcs[prop][ix]


    def _check_prop(self, prop:property | str) -> str:
        if isinstance(prop, property):
            prop = prop.__name__
        if not isinstance(prop, str):
            raise ValueError(
                '`prop` has to be of type `property` or `str`!'
            )
        if not prop in self.ADD_DATA_UI_PROPS:
            raise ValueError(
                'Provided `prop` not available as property in `ProjectData`'
            )
        return prop
    
    @staticmethod
    def _loop_funcs(funcs, arg):
        for func in funcs:
            func(arg)
