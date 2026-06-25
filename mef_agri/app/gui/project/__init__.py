from ....data.project import ProjectData


class ProjectDataGUI(ProjectData):
    ADD_DATA_UI_PROPS = [
        'processed_field', 'processed_interface', 'progress', 'add_data_error', 
        'add_data_success'
    ]
    
    def __init__(self, project_dir, gpkg_name):
        super().__init__(project_dir, gpkg_name)

    @ProjectData.processed_field.setter
    def processed_field(self, fname):
        self._pfld = fname

    @ProjectData.processed_interface.setter
    def processed_interface(self, iname):
        self._pintf = iname

    @ProjectData.progress.setter
    def progress(self, pstate):
        self._prgr = pstate

    @ProjectData.add_data_error.setter
    def add_data_error(self, err):
        self._aderr = err

    @ProjectData.add_data_success.setter
    def add_data_success(self, succ):
        self._adsucc = succ

    def add_data_interaction(self, prop:property | str, func, obj=None):
        """
        Register handlers if one of the following properties is changed

        * :func:`processed_field`
        * :func:`processed_interface`
        * :func:`progress`
        * :func:`add_data_error`
        * :func:`add_data_success`

        ``prop`` can be provided as string (e.g. ``'progress'``) or as 
        property (e.g. ``ProjectData.progress``).

        :param prop: specify at which property-change ``func`` will be called
        :type prop: property | str
        :param func: function which should be called when ``prop`` changes
        :type func: function or method
        :param obj: object reference if ``func`` is a method, defaults to None
        :type obj: object, optional
        """
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
        if not prop in self._gui_funcs.keys():
            self._gui_funcs[prop] = []
        self._gui_funcs[prop].append({'func': func, 'obj': obj})