import os
import pandas as pd
import numpy as np

from mef_agri.data.interface import Interface


class MyInterface(Interface):
    def __init__(self, temp:str, nloops:int=int(9e6), nrand:int=int(1e4)):
        super().__init__()
        self._nloops = nloops
        self._nrand = nrand
        self._temp = temp
        self.n_processes = 10

    @Interface.add_data_task(order=1)
    def step_01(self):
        data = np.random.randint(1, 100, self._nrand)
        ndata = self._nrand // self.n_processes
        ret = {}
        for i in range(self.n_processes):
            fp = os.path.join(self._temp, 'data-{}.txt'.format(i + 1))
            if i + 1 == self.n_processes:
                sli = slice(i * ndata, data.shape[0])
            else:
                sli = slice(i * ndata, (i + 1) * ndata)
            fio = open(fp, 'w')
            np.savetxt(fp, data[sli])
            fio.close()
            ret[self.process_ids[i]] = {
                'data_file': fp, 'nloops': self._nloops
            }
        return ret
    
    @Interface.add_data_task(order=2, parallel=True)
    def task_01(self, **kwargs):
        from numpy import atleast_2d, loadtxt

        pid, queue = kwargs['pid'], kwargs['queue']
        arr = atleast_2d(loadtxt(kwargs['data_file']))
        counter = int(1e6)
        ressum = 0
        for i in range(kwargs['nloops']):
            res = (arr @ arr.T)[0, 0] / arr.shape[0]
            ressum += res 
            if (i % counter) == 0:
                msg = 'Process with id `{}` >>> '
                msg += 'processed {}-times with current result = {}'
                queue.put(('__MESSAGE__', msg.format(pid, i, res)))
        
        queue.put(('res_{}'.format(pid), ressum))

    @Interface.add_data_task(order=3)
    def step_02(self, **kwargs):
        print(kwargs)


if __name__ == '__main__':
    wdir = os.path.split(__file__)[0]
    intf = MyInterface(wdir)
    intf.prj_add_data()
