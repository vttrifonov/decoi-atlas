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

        model1 = lambda d, s: fit(R.fit, d, s, '~group_per_sample')

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
    def samples2(self):
        x = self.data2.drop_dims(['feature_id', 'group_id1'])
        return x

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
    a = self.data2[['blueprint.labels', 'purification']].to_dataframe().drop_duplicates()
    for _, x in a.iterrows():
        print(
            x['purification'], x['blueprint.labels'], 
            self.rnaseq_counts2(x['purification'], x['blueprint.labels']).sizes
        )
    
    # %%
    def cor(x1, x2, dim):
        x3 = x1*x2
        x4 = ~x3.isnull()
        x4 = x4/x4

        x1 = x1*x4
        x1 = x1 - x1.mean(dim=dim)
        x1 /= np.sqrt((x1**2).sum(dim=dim))
        x2 = x2*x4
        x2 = x2 - x2.mean(dim=dim)
        x2 /= np.sqrt((x2**2).sum(dim=dim))
        x3 = (x1*x2).sum(dim=dim)
        x3 = x3.rename('cor')
        return x3    

    # %%
    x2 = self.model1.copy()
    x2 = x2.drop_dims('group_id2')
    x2 = x2.rename(group_id4='group_id')
    x2['logit_p_value'] = np.log2(x2.p/(1-x2.p))

    # %%
    x3 = x2.coef.sel(var='case')
    x4 = cor(
        x3,
        x3.rename(group_id='group_id1'),
        'feature_id'
    )

    # %%
    from plotnine import *

    x5 = x4.to_dataframe().reset_index()
    x5 = x5[~x5.cor.isna()]

    # %%
    (
        ggplot(x5)+aes('cor')+
            geom_freqpoly(bins=30)
    )
    

    # %%
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    x6 = (1-x4.data)/2
    x6 = x6 - np.diag(np.diag(x6))
    x6 = squareform(x6)
    x6 = linkage(x6)
    dendrogram(x6, labels=x4.group_id.data, color_threshold=0.17)
    x6 = dendrogram(x6, no_plot=True, labels=x4.group_id.data, color_threshold=0.17)
    x6 = pd.DataFrame(dict(
        group_id=x6['ivl'],
        ord=range(len(x6['ivl'])),            
        color=x6['leaves_color_list']
    ))
    x6 = x6.merge(
        x2[['purification', 'source', 'blueprint.labels']].
            to_dataframe().reset_index(),        
    )
    x6_1 = x6.rename(columns={
        k: k+'1'
        for k in x6.columns    
    })

    # %%
    x7 = x6.sort_values('ord').group_id.to_list()
    for x in ['group_id', 'group_id1']:
        x5[x] = x5[x].astype(pd.CategoricalDtype(x7))

    # %%
    (
        ggplot(x5)+
            aes('group_id', 'group_id1', fill='cor')+
            geom_tile()+
            scale_fill_gradient2(
                low='blue', mid='white', high='red',
                midpoint=0
            )+
            theme(axis_text=element_text(size=6))
    )

    # %%
    x6.sort_values('ord')

    # %%
    x4 = xa.merge([
        x4, 
        x6.set_index('group_id').to_xarray(),
        x6_1.set_index('group_id1').to_xarray()
    ])

    # %%
    x9 = x4.to_dataframe().reset_index()
    x9 = x9[x9.source=='model1_control_mild']
    x9 = x9[x9.purification=='FreshEryLysis']
    x9 = x9[x9.source1=='model1_control_mild']
    x9 = x9[x9.purification1=='FreshPBMC']
    for x, y in [('blueprint.labels', 'ord'), ('blueprint.labels1', 'ord1')]:
        x9[x] = x9[x].astype(pd.CategoricalDtype(
            x9[[x,y]].drop_duplicates().sort_values(y)[x].to_list()
        ))
    (
        ggplot(x9)+
            aes('blueprint.labels', 'blueprint.labels1', fill='cor')+
            geom_tile()+
            scale_fill_gradient2(
                low='blue', mid='white', high='red',
                midpoint=0
            )+
            theme(axis_text=element_text(size=6))
    )

    # %%
    x9 = x4.to_dataframe().reset_index()
    x9 = x9[x9.group_id!=x9.group_id1].copy()
    x9['g'] = np.apply_along_axis(
        lambda x: np.array('\n'.join(np.sort(x)), dtype=object),
        1,
        x9[['color', 'color1']].to_numpy()
    )
    (
        ggplot(x9)+
            aes('g', 'cor')+
            geom_violin()+geom_point()+
            theme(
                axis_text_x=element_text(angle=45, hjust=1)
            )
    )

    # %%
    x10 = [26, 44]
    x10 = x2.sel(var='case', group_id=x10).\
        logit_p_value.\
        to_dataset(dim='group_id').\
        rename(dict(zip(x10, ['x', 'y']))).\
        to_dataframe().reset_index()
    
    (
        ggplot(x10)+
            aes('x', 'y')+
            geom_point()
    )
    
    # %%
    x11 = x2.sel(var='case')
    #x11 = x11.sel(group_id=x11['blueprint.labels']=='Neutrophils')
    #x11 = x11.sel(group_id=x11.purification=='FreshEryLysis')
    x11 = x11.sel(group_id=x11.source=='model1_control_severe')
    x11 = x11.to_dataframe().reset_index()
    x11 = x11[x11.p<1e-4]
    (
        ggplot(x11)+
            aes('np.clip(coef, -5, 5)', 'np.clip(-np.log10(p), 0, 5)')+
            geom_point()
    )

    # %%
    x12 = x2.sel(var='case')
    #x12 = x12.sel(group_id=x12.source=='model1_control_mild')
    x12 = x12.sel(group_id=x12['blueprint.labels']=='Monocytes')
    x12 = x12.to_dataframe().reset_index()
    x12 = x12[x12.p<1e-4]
    x12 = x12.groupby(['purification', 'blueprint.labels', 'source']).size()
    x12 = x12.sort_values()
    x12

    # %%
    from rpy2.robjects import r as R
    R.setwd(str(config.root))
    R.source('.Rprofile')
    R.source('deg/limma.R')

    # %%
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter
    from ..rpy_conversions import xa2ri, dict2ri

    # %%
    x = np.arange(4).reshape((2,2))
    x = xa.DataArray(x, coords=[('a', [1,2]), ('b', ['a', 'b'])])
    
    with localconverter(ro.default_converter+xa2ri):
        R.f1(x)

# %%
    with localconverter(ro.default_converter+xa2ri) as co:
        print(R.f2())

    # %%



# %%
