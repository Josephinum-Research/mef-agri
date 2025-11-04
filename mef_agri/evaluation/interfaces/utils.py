from datetime import date

from ..evaluation.stats_utils import DISTRIBUTIONS


def nn_params_dict2model(nnps:dict, epoch:date=None) -> dict:
    def get_distr_param(dentry, i, j=None):
        if isinstance(dentry, list):
            if j is not None:
                return dentry[i][j]
            else:
                return dentry[i]
        else:
            return dentry

    ret = {}
    for lname, ldict in nnps.items():
        _, lorder = lname.split('_')

        # bias values
        bvals = ldict['bias']['values']
        for i, bias in zip(range(len(bvals)), bvals):
            bname = 'l{}_b{}'.format(lorder, i + 1)
            dinfo = ldict['bias']['distr']
            did = get_distr_param(dinfo['distr_id'], i)
            std = get_distr_param(dinfo['std'], i)
            ret[bname] = {'value': bias, 'distr': {'distr_id': did, 'std': std}}

            # further distribution parameters if necessary
            if did in (DISTRIBUTIONS.TRUNCNORM_1D, DISTRIBUTIONS.CATEGORICAL_1D):
                ret[bname]['distr']['ub'] = get_distr_param(dinfo['ub'], i)
                ret[bname]['distr']['lb'] = get_distr_param(dinfo['lb'], i)

            if epoch is not None:
                ret[bname]['epoch'] = epoch

        # affine transformation values
        atmrows = ldict['affine_transform']['values']
        for i, row in zip(range(len(atmrows)), atmrows):
            for j, wij in zip(range(len(row)), row):
                wname = 'l{}_w{}_{}'.format(lorder, i + 1, j + 1)
                dinfo = ldict['affine_transform']['distr']
                did = get_distr_param(dinfo['distr_id'], i, j=j)
                std = get_distr_param(dinfo['std'], i, j=j)
                ret[wname] = {
                    'value': wij, 'distr': {'distr_id': did, 'std': std}
                }

                # further distribution parameters if necessary
                if did in (DISTRIBUTIONS.TRUNCNORM_1D, DISTRIBUTIONS.CATEGORICAL_1D):
                    ret[wname]['distr']['ub'] = get_distr_param(
                        dinfo['ub'], i, j=j
                    )
                    ret[wname]['distr']['lb'] = get_distr_param(
                        dinfo['lb'], i, j=j
                    )

                if epoch is not None:
                    ret[wname]['epoch'] = epoch

    return ret
