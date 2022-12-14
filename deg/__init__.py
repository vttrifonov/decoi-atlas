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
    if group_id is None:
        i = 0
        while 'dim_'+str(i) in x1:
            i = i + 1
        group_id = 'dim_'+str(i)
    
    _group_id = '_'+group_id
    x1 = x1.to_dataframe()
    x2 = x1.drop_duplicates().copy()
    x2[group_id] = range(x2.shape[0])
    x1 = x1.reset_index().\
        merge(x2)[[group_id]+x1.index.names].\
        set_index(x1.index.names).to_xarray().\
        rename({group_id: _group_id})
    x2 = x2.set_index(group_id).to_xarray()
    x1 = xa.merge([x1, x2], join='inner')
    if f is None:
        return x1

    x3 = x3.groupby(x1[_group_id]).apply(f)
    x3 = xa.merge([
        x1,
        x3.rename({_group_id: group_id})
    ], join='inner')
    return x3

# %%
def r_func(f):
    from rpy2.robjects.conversion import localconverter
    from ..rpy_conversions import num2ri, dict2ri

    def wrap(*args):
        with localconverter(num2ri+dict2ri):
            return f(*args)
    return wrap

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
        x1 = x1.drop('_group_id1')

        x2 = x1.drop_dims(['cell_id', 'feature_id'])\
            .drop('blueprint.labels')
        x2 = groupby(
            x2.drop('n'), 'group_id2', 
            x2.n, lambda x: x.sum(dim='group_id1')
        )
        x2 = x2.n[x2._group_id2].drop('group_id2')
        x1['freq'] = x1.n/x2

        x3 = x1.drop_dims('cell_id').\
            drop(['experiment', 'disease_stage', 'n'])
        x3 = groupby(
            x3.drop_dims('feature_id').drop('freq'), 
            'group_id2',
            x3[['counts', 'freq']],
            lambda x: x.mean(dim='group_id1')
        )

        return x3

    @compose(property, lazy, XArrayCache())
    def model1(self):
        from rpy2.robjects import r as R
        R.setwd(str(config.root))
        R.source('.Rprofile')
        R.source('deg/limma.R')

        model1 = lambda d, s: r_func(R.fit)(d, s, '~group_per_sample')

        x1 = self.data2.copy()
        x1 = x1.drop_dims('group_id1')
        x1['counts'] = x1.counts.transpose('feature_id', 'group_id2')
        x8 = [['control', 'mild'], ['control', 'severe'], ['mild', 'severe']]        

        def fit1(x9, i):
            x9 = x9.sel(group_id3=x9._group_id3.data[0])
            print(
                x9['purification'].item(),
                x9['blueprint.labels'].item(),
            )
            x11 = list()
            for x7 in x8:
                print(x7)
                x4 = x9.sel(group_id2=x9.group_per_sample.isin(x7))                
                x6 = x4.group_per_sample.to_dataframe()
                if x6.shape[0]==0:
                    continue
                x12 = x6.groupby('group_per_sample').size()
                if x12.shape[0]<len(x7) or any(x12==1):
                    continue
                x6['group_per_sample'] = x6.group_per_sample.astype(pd.CategoricalDtype(x7))
                x10 = model1(x4.counts, x6).drop_dims('group_id2')
                x10['var'] = 'var', ['control', 'case']
                x10['group_id4'] = 'group_id4', [i]
                i = i + 1
                x10['control'] = 'group_id4', [x7[0]]
                x10['case'] = 'group_id4', [x7[1]]
                x10['source'] = 'group_id4', ['model1_' + '_'.join(x7)]
                x11.append(x10)
            if len(x11)==0:
                return None, i
            x11 = xa.concat(x11, dim='group_id4')
            x9 = x9[['purification', 'blueprint.labels', 'group_id3']].\
                reset_coords().\
                expand_dims(group_id4=x11.group_id4.data)
            x11 = xa.merge([x11, x9])
            return x11, i

        x3 = x1.drop_dims('feature_id').drop(['freq', 'group_per_sample', 'donor'])    
        x3 = groupby(x3, 'group_id3')
        x3 = xa.merge([x1[['counts', 'group_per_sample']], x3], join='inner')

        x4 = list()
        i = 0
        for _, x9 in list(x3.groupby('_group_id3')):
            x5, i = fit1(x9, i)
            x4.append(x5)
        x4 = [x for x in x4 if x is not None]
        x4 = xa.concat(x4, 'group_id4')
        x4 = xa.merge([x4, x3._group_id3], join='inner')
        
        return x4

    @compose(property, lazy, XArrayCache())
    def voom1(self):
        from rpy2.robjects import r as R
        R.setwd(str(config.root))
        R.source('.Rprofile')
        R.source('deg/limma.R')

        x1 = self.data2.copy()
        x1 = x1.drop_dims('group_id1')
        x1['counts'] = x1.counts.transpose('feature_id', 'group_id2')

        x3 = x1.drop_dims('feature_id').drop(['freq', 'group_per_sample', 'donor']) 
        x3 = groupby(x3, 'group_id3')
        x3 = xa.merge([x1[['counts', 'group_per_sample']], x3], join='inner')

        x4 = list()
        #x3 = x3.sel(group_id2=x3._group_id3.isin([18]))
        for group_id3, x9 in x3.groupby('_group_id3'):
            # group_id3, x9 = next(iter(x3.groupby('_group_id3')))
            print(group_id3)

            x6 = x9.group_per_sample.to_dataframe()
            if x6.group_per_sample.drop_duplicates().shape[0]==1:
                continue
            x6['group_per_sample'] = x6.group_per_sample.astype(pd.CategoricalDtype(
                ['control', 'mild', 'severe']
            ))

            x10 = r_func(R.fit3)(x9.counts, x6, '~0+group_per_sample')
            x10 = xa.merge([v.rename(k) for k, v in x10.items()])
            x10['var'] = 'var', ['control', 'mild', 'severe']
            x10['var1'] = 'var1', x10['var'].data
            x10 = x10.expand_dims(group_id3=[group_id3])
            x4.append(x10)
        x4 = xa.concat(x4, 'group_id3')
        x4 = xa.merge([
            x4, 
            x3[['purification', 'blueprint.labels', '_group_id3']]
        ], join='inner')
        return x4

    @compose(property, lazy, XArrayCache())
    def model2(self):
        from rpy2.robjects import r as R
        R.setwd(str(config.root))
        R.source('.Rprofile')
        R.source('deg/limma.R')

        x1 = self.voom1.copy()
        x8 = [['control', 'mild'], ['control', 'severe'], ['mild', 'severe']]        

        i = 0
        x4 = list()
        #x1 = x1.sel(group_id3=[0])
        for group_id3, x9 in x1.groupby('group_id3'):
            #group_id3, x9 = next(iter(x1.groupby('group_id3')))
            print(group_id3)
            for x7 in x8:
                #x7 = x8[0]
                print(x7)
                x10 = xa.DataArray(
                    np.zeros((3,2)),
                    (x1['var'], ('var2', ['control', 'case']))
                )
                x10.loc[x7, :] = [[1,-1],[0,1]]
                x10 = r_func(R.fit4)(
                    *list(x9[['coef', 'cov', 'std', 'sigma', 'df']].values()),
                    x10
                )
                x10 = xa.merge([v.rename(k) for k, v in x10.items()])
                x10 = x10.rename(var2='var')
                x10 = x10.expand_dims(group_id4=[i])
                x10['control'] = 'group_id4', [x7[0]]
                x10['case'] = 'group_id4', [x7[1]]
                x10['_group_id3'] = 'group_id4', [group_id3]
                x4.append(x10)
                i = i + 1
        x4 = xa.concat(x4, 'group_id4')
        x4 = xa.merge([
            x4, 
            x1[['purification', 'blueprint.labels']]
        ], join='inner')
        
        return x4

    @compose(property, lazy, XArrayCache())
    def voom2(self):
        from rpy2.robjects import r as R
        R.setwd(str(config.root))
        R.source('.Rprofile')
        R.source('deg/limma.R')

        x1 = self.data2.copy()
        x1 = x1.drop_dims('group_id1')
        x1['counts'] = x1.counts.transpose('feature_id', 'group_id2')

        x3 = x1.drop_dims('feature_id').drop(['freq', 'group_per_sample', 'donor']) 
        x3 = groupby(x3, 'group_id3')
        x3 = xa.merge([x1[['counts', 'group_per_sample']], x3], join='inner')

        #x5 = x3[['_group_id3', 'group_per_sample']].to_dataframe().reset_index()
        #x6 = x5._group_id3.astype(str) + x5.group_per_sample
        #x6 = x5.groupby(x6).group_id2.transform(len)
        #x5 = x5[x6>1]
        #x3 = x3.sel(group_id2=x5.group_id2.to_list())

        x2 = x3[['_group_id3', 'group_per_sample']].to_dataframe()
        x2['var'] = x3._group_id3.astype(str) + x2.group_per_sample
        x2['var'] = x2['var'].astype('category').cat.codes.astype(str)
        x4 = r_func(R.fit3)(x3.counts, x2[['var']], '~0+var')
        x4 = xa.merge([v.rename(k) for k, v in x4.items()])
        x4['var'] = 'var', x4['var'].to_series().str.replace('var', '').astype(int)
        x4['var1'] = 'var1', x4['var1'].to_series().str.replace('var', '').astype(int)
        x2['var'] = x2['var'].astype(int)    
        x6 = x2.drop_duplicates().set_index('var').\
            to_xarray().rename(_group_id3='var_group_id3')
        x7 = xa.merge([
            x4, x6, 
            x3[['_group_id3', 'purification', 'blueprint.labels']]
        ])
        x7 = x7.sel(var1=x7['var'].data)
        return x7

    @compose(property, lazy, XArrayCache())
    def model3(self):
        from rpy2.robjects import r as R
        R.setwd(str(config.root))
        R.source('.Rprofile')
        R.source('deg/limma.R')

        x1 = self.voom2.copy()
        x2 = [['control', 'mild'], ['control', 'severe'], ['mild', 'severe']]        

        i = 0
        x9 = []        
        x3 = x1[['var_group_id3', 'group_per_sample']].to_dataframe()
        #x3 = x3[x3.var_group_id3<=10]
        #x4 = x3.groupby('var_group_id3')
        for group_id3, x5 in x3.groupby('var_group_id3'):
            #group_id3, x5 = next(iter(x3.groupby('var_group_id3')))
            print(group_id3)            
            x5 = x5.reset_index().set_index('group_per_sample')['var']
            for x6 in x2:                
                #x6 = x2[1]
                print(x6)
                x8 = [x5.get(x, None) for x in x6]
                if any(x is None for x in x8):
                    continue

                x7 = xa.DataArray(
                    np.zeros((x1.sizes['var'], 2)),
                    [x1['var'], ('var2', ['control', 'case'])]
                )
                x7.loc[x8, :] = [[1,-1], [0,1]]
                x7 = r_func(R.fit4)(
                    *list(x1[['coef', 'cov', 'std', 'sigma', 'df']].values()),
                    x7
                )
                x7 = xa.merge([v.rename(k) for k, v in x7.items()])
                x7 = x7.rename(var2='var')
                x7 = x7.expand_dims(group_id4=[i])
                x7['control'] = 'group_id4', [x6[0]]
                x7['case'] = 'group_id4', [x6[1]]
                x7['_group_id3'] = 'group_id4', [group_id3]
                x9.append(x7)
                i = i + 1        
        x9 = xa.concat(x9, dim='group_id4')
        x9 = xa.merge([
            x9, 
            x1[['purification', 'blueprint.labels']]
        ], join='inner')
        
        return x9

    @compose(property, lazy, XArrayCache())
    def samples2(self):
        x = self.data2.drop_dims(['feature_id', 'group_id1'])
        return x

    @compose(property, lazy, XArrayCache())
    def genes2(self):
        return self.data2.feature_id.rename('symbol')

    def rnaseq_counts2(self, purification, cell_type):  
        analysis = self
        class counts:      
            storage = analysis.storage/'rnaseq_counts2'/(purification+'_'+cell_type)

            @compose(property, lazy, XArrayCache())
            def data(self):
                import sparse
                x = analysis.data2.drop_dims('group_id1')
                x = x.sel(group_id2=x['purification']==purification)
                x = x.sel(
                    group_id2=x['blueprint.labels']==cell_type
                )
                x = x['counts']
                x.data = sparse.COO(x.data)
                return x

        counts = counts()

        return counts.data.todense()

analysis = _analysis()

# %%
if __name__ == '__main__':
    self = analysis   

    # %%