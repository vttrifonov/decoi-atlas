# %%
if __name__ == '__main__':
    import os
    import sys
    sys.path[0] = os.path.expanduser('~/projects/')
    __package__ = 'decoi_atlas.deg'

# %%
import numpy as np
import xarray as xa
import pandas as pd

from ..common import helpers
from ..common.caching import compose, lazy, XArrayCache
from .._data import data
from .._helpers import config

# %%
def groupby(x1, group_id='group_id', x3 = None, f = None):
    x1 = x1.to_dataframe()
    x2 = x1.drop_duplicates().\
        assign(**{group_id: lambda x: range(x.shape[0])})
    x1 = x1.reset_index().merge(x2)[[group_id]+x1.index.names].\
        set_index(x1.index.names).to_xarray().\
        rename({group_id: '_sample_group_id'})
    x2 = x2.set_index(group_id).to_xarray()
    x1 = xa.merge([x1, x2], join='inner')
    if f is None:
        return x1

    x3 = xa.merge([x3, x1._sample_group_id]).groupby('_sample_group_id').apply(f)
    x3 = xa.merge([
        x1,
        x3.rename({'_sample_group_id': group_id})
    ])
    return x3

# %%
def fit(r, d, s, f):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from ..rpy_conversions import xa2ri, dict2ri

    with localconverter(ro.default_converter+xa2ri) as co:
        x1 = co.py2rpy(d)
    with localconverter(ro.default_converter+pandas2ri.converter) as co:
        x2 = co.py2rpy(s)
    with localconverter(ro.default_converter+xa2ri+dict2ri) as co:
        x5 = co.rpy2py(r(x1, x2, f))

    x5 = [v.rename(k) for k, v in x5.items()]
    x5 = xa.merge(x5, join='inner')

    return x5

# %%
class _analysis:    
    storage = config.cache/'deg'

    @compose(property, lazy, XArrayCache())
    def data1(self):
        x = data.c2_wb_pbmc
        x = x.sel(cell_id=x.cell_donor!='')
        x = x[[
            'cell_donor', 'cell_purification', 'cell_experiment',
            'cell_group_per_sample', 'cell_disease_stage', 
            'cell_blueprint.labels', 'counts'
        ]]

        x1 = x.drop_dims('feature_id')
        x1 = x1.rename({k: k.replace('cell_', '') for k in x1.keys()})
        x1 = groupby(
            x1, 'group_id1',
            x.counts,
            lambda x: xa.merge([
                x.sum(dim='cell_id').todense(),
                (x**2).sum(dim='cell_id').todense().rename('counts2'),
                xa.DataArray(x.sizes['cell_id'], name='n')
            ])
        )

        return x1

    @compose(property, lazy)
    def data2(self):
        x1 = self.data1.copy()
        x1 = x1.rename(_sample_group_id='cell_group_id1')

        x2 = x1.drop_dims(['cell_id', 'feature_id']).drop('blueprint.labels')
        x2 = groupby(x2.drop('n'), 'group_id2', x2.n, lambda x: x.sum(dim='group_id1'))
        x1['freq'] = x1.n/x2.n[x2._sample_group_id].drop('group_id2')

        x3 = x1.drop_dims('cell_id').drop(['experiment', 'disease_stage', 'n'])
        x3 = groupby(
            x3.drop_dims('feature_id').drop('freq'), 
            'group_id2',
            x3[['counts', 'freq']],
            lambda x: x.mean(dim='group_id1'),
        )

        return x3

    @compose(property, lazy, XArrayCache())
    def model1(self):
        from rpy2.robjects import r as R
        R.setwd(str(config.root))
        R.source('.Rprofile')
        R.source('deg/limma.R')

        model1 = lambda d, s: fit(R.fit, d, s, '~group_per_sample')

        x1 = self.data2.copy()
        x1['counts'] = x1.counts.transpose('feature_id', 'group_id2')
        x8 = [['control', 'mild'], ['control', 'severe'], ['mild', 'severe']]

        def fit1(x9):
            x9 = x9.sel(group_id3=x9._sample_group_id.data[0])
            print(
                x9['purification'].item(),
                x9['blueprint.labels'].item()
            )
            x9 = x9.drop('group_id3')
            x11 = list()
            for x7 in x8:  
                print(x7)
                x4 = x9.sel(group_id2=x9.group_per_sample.isin(x7))
                x6 = x4.group_per_sample.to_dataframe()
                x6['group_per_sample'] = x6.group_per_sample.astype(pd.CategoricalDtype(x7))
                x10 = model1(x4.counts, x6).drop_dims('group_id2')
                x10 = x10.expand_dims(source=['model1_' + '_'.join(x7)])
                x11.append(x10)
            x11 = xa.concat(x11, dim='source')
            return x11
        
        x3 = x1.drop_dims(['feature_id', 'group_id1']).drop(['freq', 'group_per_sample', 'donor'])
        x3 = groupby(x3, 'group_id3')
        x3 = xa.merge([x1[['counts', 'group_per_sample']], x3], join='inner')
        x4 = x3.groupby('_sample_group_id').apply(fit1)
        x4 = x4.rename(_sample_group_id='group_id3')
        x4 = xa.merge([x4, x3.drop(['counts', 'group_per_sample'])], join='inner')
        
        return x4

analysis = _analysis()

# %%
if __name__ == '__main__':
    self = analysis

    # %%
    self.model1


# %%
