import json

class Crops:
    class Maize:
        NAME = 'maize'
        class Cultivars:
            class Generic:
                NAME = 'generic'
                PARAMS = {}

    class Soybean:
        NAME = 'soybean'
        class Cultivars:
            class Generic:
                NAME = 'generic'
                PARAMS = {}

    class WinterWheat:
        NAME = 'winter_wheat'
        class Cultivars:
            class Generic:
                NAME = 'generic'
                PARAMS = {}

    class WinterBarley:
        NAME = 'winter_barley'
        class Cultivars:
            class Generic:
                NAME = 'generic'
                PARAMS = {}


class DBIntegration(object):
    COL_CROP = 'crop'
    COL_CULTIVAR = 'cultivar'
    COL_PARAMS = 'parameters'

    def __init__(self, table_name:str):
        self._tname:str = table_name

    @property
    def sql_table_exists(self) -> str:
        sql = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND '
        sql += f'name=\'{self._tname}\';'
        return sql

    @property
    def sql_create(self) -> str:
        sql = f'CREATE TABLE {self._tname} ({self.COL_CROP} TEXT, '
        sql += f'{self.COL_CULTIVAR} TEXT, {self.COL_PARAMS} TEXT, '
        sql += f'PRIMARY KEY ({self.COL_CROP}, {self.COL_CULTIVAR}));'
        return sql
    
    @property
    def sql_insert_defaults(self) -> str:
        sql = f'INSERT INTO {self._tname} '
        sql += f'({self.COL_CROP}, {self.COL_CULTIVAR}, {self.COL_PARAMS}) '
        sql += 'VALUES '
        for cult in self.default_cultivars():
            sql += self.sql_insert_tuple(
                cult['crop'], cult['cultivar'], cult['params']
            )
            sql += ', '
        return sql[:-2] + ';'
    
    @property
    def sql_query_all(self) -> str:
        sql = f'SELECT * FROM {self._tname};'
        return sql

    def sql_insert(self, crop:str, cult:str, params:str | dict) -> str:
        sql = f'INSERT INTO {self._tname} '
        sql += f'({self.COL_CROP}, {self.COL_CULTIVAR}, {self.COL_PARAMS}) '
        sql += 'VALUES ' + self.sql_insert_tuple(crop, cult, params) + ';'
        return sql

    def sql_insert_tuple(self, crop:str, cult:str, params:str | dict) -> str:
        if isinstance(params, str):
            try:
                json.loads(params)
            except:
                msg = 'Provided parameters cannot be parsed to dictionary!'
                raise ValueError(msg)
        else:
            params = json.dumps(params)
        return f'(\'{crop}\', \'{cult}\', \'{params}\')'
    
    def default_cultivars(self) -> list[dict]:
        return [
            {
                'crop': Crops.WinterWheat.NAME,
                'cultivar': Crops.WinterWheat.Cultivars.Generic.NAME,
                'params': Crops.WinterWheat.Cultivars.Generic.PARAMS
            },
            {
                'crop': Crops.WinterBarley.NAME,
                'cultivar': Crops.WinterBarley.Cultivars.Generic.NAME,
                'params': Crops.WinterBarley.Cultivars.Generic.PARAMS
            },
            {
                'crop': Crops.Maize.NAME,
                'cultivar': Crops.Maize.Cultivars.Generic.NAME,
                'params': Crops.Maize.Cultivars.Generic.PARAMS
            },
            {
                'crop': Crops.Soybean.NAME,
                'cultivar': Crops.Soybean.Cultivars.Generic.NAME,
                'params': Crops.Soybean.Cultivars.Generic.PARAMS
            }
        ]
