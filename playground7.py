# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
import matplotlib.pyplot as plt
import statsmodels.api as sm

from .common import helpers
from .common.caching import compose, lazy, XArrayCache, CSVCache
from ._data import data
from ._helpers import config, plot_table, quantile

storage = config.cache/'playground7'

# %%
x1 = data.c1_pbmc
x1 = x1.rename(cell_group_per_sample='cell_diagnosis')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis']]
x1 = x1.to_dataframe()
x1 = sm.stats.Table.from_data(x1)
print(plot_table(x1))

# %%
x1 = data.c1_pbmc
x1 = x1.rename(cell_group_per_sample='cell_diagnosis')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']]
x1 = x1.to_dataframe()
x2 = x1.groupby(['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']).size()
x3 = x1.groupby(['cell_diagnosis', 'cell_donor']).size()
x4 = pd.merge(
    x2.rename('n').reset_index(), 
    x3.rename('m').reset_index()
)
x4['freq'] = x4.n/x4.m
x4 = x4[x4['cell_blueprint.labels']=='NK cells']
x4 = x4[x4['cell_diagnosis']!='']
print(
    ggplot(x4)+aes('cell_diagnosis', '100*freq')+
        geom_violin(aes(fill='cell_diagnosis'))+
        geom_boxplot(aes(fill='cell_diagnosis'), width=0.2)+
        geom_jitter(width=0.1)+
        labs(x='', y='NK cells frequency (%)')+
        theme(
            figure_size=(4, 3),
            legend_position='none'
        ) 
)

# %%
x1 = data.c2_wb_pbmc
x1 = x1.sel(cell_id=x1.cell_purification=='FreshPBMC')
x1 = x1.sel(cell_id=x1.cell_diagnosis!='')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis']]
x1 = x1.to_dataframe()
x1 = sm.stats.Table.from_data(x1)
print(plot_table(x1))

# %%
x1 = data.c2_wb_pbmc
x1 = x1.sel(cell_id=x1.cell_purification=='FreshPBMC')
x1 = x1.sel(cell_id=x1.cell_diagnosis!='')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']]
x1 = x1.to_dataframe()
x2 = x1.groupby(['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']).size()
x3 = x1.groupby(['cell_diagnosis', 'cell_donor']).size()
x4 = pd.merge(
    x2.rename('n').reset_index(), 
    x3.rename('m').reset_index()
)
x4['freq'] = x4.n/x4.m
x4 = x4[x4['cell_blueprint.labels']=='NK cells']
x4 = x4[x4['cell_diagnosis']!='']
print(
    ggplot(x4)+aes('cell_diagnosis', '100*freq')+
        geom_violin(aes(fill='cell_diagnosis'))+
        geom_boxplot(aes(fill='cell_diagnosis'), width=0.2)+
        geom_jitter(width=0.1)+
        labs(x='', y='NK cells frequency (%)')+
        theme(
            figure_size=(5, 4),
            legend_position='none'
        )
)

#%% 
from types import SimpleNamespace

class _analysis1:
    @compose(property, lazy, XArrayCache())
    def feature_entrez(self):
        from .sigs.entrez import symbol_entrez

        x2 = self.x2
        x6 = self.data1[x2.feature].to_series()
        x6 = symbol_entrez(x6)
        x6 = x6.rename(
            symbol=x2.feature,
            Entrez_Gene_ID = 'entrez'
        )
        return x6.rename('feature_entrez')

    @compose(property, lazy, XArrayCache())
    def log1p_rpm(self):
        x1 = self.data1
        x2 = self.x2

        x1['group'] = (x1[x2.donor] + x1[x2.cells]).to_series().\
            astype('category').cat.codes.to_xarray()
        x1[x2.counts].data = x1[x2.counts].data.todense()
        x3 = x1.groupby('group')
        x3 = x3.apply(lambda x: xa.merge([
            x[[x2.donor, x2.diag, x2.cells]].isel({x2.cell: 0}),
            x.counts.sum(dim=x2.cell),
            xa.DataArray(x.sizes[x2.cell], name='n')
        ]))
        x3['total'] = x3['counts'].sum(dim=x2.feature)
        x3['log1p_rpm'] = np.log1p(x3['counts']/x3['total'])
        return x3

    @compose(property, lazy, XArrayCache())
    def gsva(self):
        from .sigs._sigs import sigs
        from .sigs.gsva import gsva
        x2 = self.x2
        x1 = sigs.all1
        x1 = x1.sel(sig=x1.sig_prefix.isin(["KEGG1", "HALLMARK"]))
        x1 = xa.merge([
            x1,
            self.feature_entrez.rename(entrez='gene'),
            self.log1p_rpm.log1p_rpm.rename(group='sample')
        ], join='inner')
        x1['log1p_rpm'] = xa.apply_ufunc(
            np.matmul, x1.feature_entrez, x1.log1p_rpm,
            input_core_dims=[['gene', x2.feature], [x2.feature, 'sample']],
            output_core_dims=[['gene', 'sample']]
        )
        x1['log1p_rpm'] = x1.log1p_rpm/x1.feature_entrez.sum(dim=x2.feature)
        x1['gene'] = x1.gene.astype(np.int32)

        x3 = gsva(
            x1.log1p_rpm,
            x1.set.to_series_sparse().reset_index().drop(columns=['set'])
        )
        x3 = xa.merge([x3.rename('gsva').to_dataset(), x1[['sig_prefix']]])
        x3['sample'] = x3['sample'].astype(x1.sample.dtype)
        x3 = x3.rename(sample='group')
        return x3.gsva
        
    @compose(property, lazy)
    def summary1(self):
        x2 = self.x2
        s = xa.merge([
            self.gsva,
            self.log1p_rpm[x2.diag]
        ])
        s = s.groupby(x2.diag).apply(lambda x: xa.Dataset(dict(
            mu = x.gsva.mean(dim='group'),
            sigma = x.gsva.std(dim='group'),
            n = x.sizes['group']
        )))
        return s

    @compose(property, lazy, CSVCache(ext='.csv'))
    def summary2(self):
        x7 = self.x2.summary2
        x2 = self.summary1
        x3 = x2.sel({self.x2.diag: x7[0]})
        x4 = x2.sel({self.x2.diag: x7[1]})

        x5 = x4.mu-x3.mu
        x6 = (x4.n*(x4.sigma**2)+x3.n*(x3.sigma**2))/(x4.n+x3.n)
        x5 = x5/x6
        x5 = xa.merge([x5.to_dataset(self.x2.diag), (x3.mu/x3.sigma).rename('control')])

        x5 = x5.to_dataframe().reset_index()
        x5 = x5[['sig','sig_prefix',x7[0]]+x7[1]]
        x5['score'] = x5[x7[1]].abs().max(axis=1)
        x5['sig'] = x5.sig.str.replace('^[^_]*_', '', regex=True)
        x5 = x5.sort_values('score', ascending=False)
        return x5

# %%
class _c1_pbmc_analysis1(_analysis1):
    storage = storage/'c1_pbmc'/'analysis1'

    x2 = SimpleNamespace(
        diag='cell_group_per_sample',
        cells='cell_blueprint.labels',
        NK='NK cells',
        counts='counts',
        donor='cell_donor',
        cell='cell_id',
        feature='feature_id',
        summary2=['control', ['mild', 'severe']]
    )

    @compose(property, lazy)
    def data1(self):
        x2 = self.x2
        x1 = data.c1_pbmc
        x1 = x1[[x2.counts, x2.donor, x2.diag, x2.cells]]
        x1 = x1.sel({x2.cell: x1[x2.cells]==x2.NK})
        x1 = x1.sel({x2.cell: x1[x2.diag]!=''})
        return x1

x1 = _c1_pbmc_analysis1()

# %%
x1.summary2

# %%
class _c2_wb_pbmc_analysis1(_analysis1):
    storage = storage/'c2_wb_pbmc'/'analysis1'

    x2 = SimpleNamespace(
        diag='cell_diagnosis',
        cells='cell_blueprint.labels',
        NK='NK cells',
        counts='counts',
        donor='cell_donor',
        cell='cell_id',
        feature='feature_id',
        summary2=['control', ['COVID-19, mild', 'COVID-19, severe']]
    )

    @compose(property, lazy)
    def data1(self):
        x2 = self.x2
        x1 = data.c2_wb_pbmc
        x1 = x1.sel(cell_id=x1.cell_purification=='FreshPBMC')
        x1 = x1.sel(cell_id=x1.cell_diagnosis!='')
        
        x1 = x1[[x2.counts, x2.donor, x2.diag, x2.cells]]
        x1 = x1.sel({x2.cell: x1[x2.cells]==x2.NK})
        x1 = x1.sel({x2.cell: x1[x2.diag]!=''})
        return x1

x1 = _c2_wb_pbmc_analysis1()

# %%
x1.summary2