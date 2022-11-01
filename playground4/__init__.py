# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground4'

# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *

from ..common import helpers
from ..common.caching import compose, lazy, XArrayCache, CSVCache
from .._data import data
from .._helpers import config, plot_table

# %%
storage = config.cache/'playground4'

# %%
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import beta

def ols(A, B):
    # this assumes A is full rank
    Q, R = np.linalg.qr(A)
    X = (Q.T) @ B
    SS = (X**2).sum(axis=0)
    X = np.linalg.inv(R) @ X
    return X, SS, Q.shape[1]

def anova(A1, A2, B):
    _, SS1, rk1 = ols(A1, B)
    X, SS2, rk2 = ols(A2, B)
    r2 = (SS2-SS1)/((B**2).sum(axis=0)-SS1)
    df1, df2 = rk2-rk1, B.shape[0]-rk1
    p = beta.sf(r2, df1/2, df2/2)
    return X, r2, p

# %%
def quantile(x, q=3):
    x1 = np.where(x==0, np.nan, x)
    x1 = pd.qcut(x1, q=q)
    x1 = x1.add_categories(pd.Interval(0,0))
    x1 = x1.reorder_categories(np.roll(np.array(x1.dtype.categories), 1))
    x1 = x1.fillna(pd.Interval(0,0))
    return x1

# %%
class _analysis:
    @compose(property, lazy)
    def data(self):
        x = data.c2_neutrophils_integrated
        x['rpk'] = x.counts.copy()
        x.rpk.data = x.rpk.data.asformat('coo')
        x['total'] = x.rpk.sum(dim='feature_id').todense()
        x['rpk'] = 1e4*x.rpk/x.total
        return x

    @compose(property, lazy)
    def data1(self):
        x = self.data
        x2 = x.sel(feature_id=['C5AR1'])
        x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
        x2 = x2.todense()
        x2 = xa.merge([
            x2.counts.assign_coords(
                feature_id=['counts_'+k for k in x2.feature_id.data]
            ).to_dataset(dim='feature_id'),
            x2.rpk.assign_coords(
                feature_id=['rpk_'+k for k in x2.feature_id.data]
            ).to_dataset(dim='feature_id'),
            x2.drop_dims(['feature_id', 'umap_dim']),
            x2.umap.to_dataset('umap_dim'),
        ])
        x2 = x2.to_dataframe()
        x2['C5AR1>0'] = np.where(x2.counts_C5AR1>0, 'C5AR1>0', 'C5AR1=0')
        x2['cell_integrated_snn_res.0.3'] = x2['cell_integrated_snn_res.0.3'].astype('category')
        x2['rpk_C5AR1_q3'] = np.where(x2.rpk_C5AR1==0, np.nan, x2.rpk_C5AR1)
        x2['rpk_C5AR1_q3'] = pd.qcut(x2.rpk_C5AR1_q3, q=3)
        x1 = x2['rpk_C5AR1_q3'].cat.add_categories(pd.Interval(0,0))
        x1 = x1.cat.reorder_categories(np.roll(np.array(x1.dtype.categories), 1))
        x1 = x1.fillna(pd.Interval(0,0))
        x2['rpk_C5AR1_q3'] = x1

        x2clust = x2.groupby('cell_integrated_snn_res.0.3')[['UMAP_1', 'UMAP_2']].\
            mean().reset_index()

        return (x2, x2clust)

    @compose(property, lazy)
    def clust1(self):
        x = self.data
        x = xa.merge([
            x['cell_integrated_snn_res.0.3'],
            x['umap'].to_dataset('umap_dim')
        ])
        x = x.to_dataframe().reset_index()
        x = x.groupby('cell_integrated_snn_res.0.3')[['UMAP_1', 'UMAP_2']].\
            mean().reset_index()
        return x

    def feature_data1(self, feature_id):
        x = self.data
        x2 = x.sel(feature_id=feature_id)
        x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
        x2 = x2.todense()
        x2 = xa.merge([
            x2[['counts', 'rpk']],
            x2.drop_dims(['umap_dim']),
            x2.umap.to_dataset('umap_dim'),
        ])
        x2 = x2.to_dataframe()
        x2['feature>0'] = np.where(x2.counts>0, 'feature>0', 'feature=0')
        x2['cell_integrated_snn_res.0.3'] = x2['cell_integrated_snn_res.0.3'].astype('category')
        x2['rpk_q3'] = np.where(x2.rpk==0, np.nan, x2.rpk)
        x2['rpk_q3'] = pd.qcut(x2.rpk_q3, q=3)
        x1 = x2['rpk_q3'].cat.add_categories(pd.Interval(0,0))
        x1 = x1.cat.reorder_categories(np.roll(np.array(x1.dtype.categories), 1))
        x1 = x1.fillna(pd.Interval(0,0))
        x2['rpk_q3'] = x1

        return x2
    
    @compose(property, lazy)
    def data2(self):
        x = data.c2_wb_pbmc
        x['rpk'] = x.counts.copy()
        x.rpk.data = x.rpk.data.asformat('coo')
        x['total'] = x.rpk.sum(dim='feature_id').todense()
        x['rpk'] = 1e4*x.rpk/x.total
        return x

    @compose(property, lazy)
    def data3(self):
        x = self.data2
        x2 = x.sel(feature_id=['C5AR1'])
        x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
        x2 = x2.todense()
        x2 = xa.merge([
            x2.counts.assign_coords(
                feature_id=['counts_'+k for k in x2.feature_id.data]
            ).to_dataset(dim='feature_id'),
            x2.rpk.assign_coords(
                feature_id=['rpk_'+k for k in x2.feature_id.data]
            ).to_dataset(dim='feature_id'),
            x2.drop_dims(['feature_id', 'umap_dim']),
            x2.umap.to_dataset('umap_dim'),
        ])
        x2 = x2.to_dataframe()
        x2['C5AR1>0'] = np.where(x2.counts_C5AR1>0, 'C5AR1>0', 'C5AR1=0')
        x2['cell_blueprint.labels'] = x2['cell_blueprint.labels'].astype('category')
        x2['rpk_C5AR1_q3'] = np.where(x2.rpk_C5AR1==0, np.nan, x2.rpk_C5AR1)
        x2['rpk_C5AR1_q3'] = pd.qcut(x2.rpk_C5AR1_q3, q=3)
        x1 = x2['rpk_C5AR1_q3'].cat.add_categories(pd.Interval(0,0))
        x1 = x1.cat.reorder_categories(np.roll(np.array(x1.dtype.categories), 1))
        x1 = x1.fillna(pd.Interval(0,0))
        x2['rpk_C5AR1_q3'] = x1

        x2clust = x2.groupby('cell_blueprint.labels')[['UMAP_1', 'UMAP_2']].\
            mean().reset_index()

        return x2, x2clust

analysis = _analysis()

self = analysis

# %%
class _c5ar1_analysis:
    @property
    def data(self):
        x = analysis.data
        x2 = x.sel(feature_id='C5AR1').drop('feature_id').todense()
        x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
        x2['rpk'] = np.log1p(x2.rpk)
        x2 = xa.merge([
            x2.cell_diagnosis,
            x2['cell_integrated_snn_res.0.3'],
            x2[['rpk']].assign(
                const=('cell_id', [1]*x2.sizes['cell_id'])
            ).to_array('coef').rename('c5ar1')
        ])

        x3 = np.log1p(x.rpk).copy()
        x3.data = x3.data.asformat('gcxs', compressed_axes=(0,))
        x2 = xa.merge([x2, x3], join='inner')

        return x2

    @compose(property, lazy, XArrayCache())
    def model(self):
        def _(x2):
            x4 = xa.apply_ufunc(
                    anova, xa.DataArray(1, [x2.cell_id, ('coef1', [0])]), x2.c5ar1, x2.rpk,
                    input_core_dims=[['cell_id', 'coef1'], ['cell_id', 'coef'], ['cell_id', 'feature_id']],
                    output_core_dims=[['coef', 'feature_id']]+[['feature_id']]*2
            )
            x4 = [x.rename(k) for x, k in zip(x4, ['X', 'r2', 'p'])]
            x4 = xa.merge(x4)
            x5 = x4.p.data
            x5[~np.isnan(x5)] = multipletests(x5[~np.isnan(x5)], method='fdr_bh')[1]
            x4['q'] = 'feature_id', x5
            return x4
        x4 = self.data.groupby('cell_diagnosis').apply(_)
        return x4

    @compose(property, lazy, XArrayCache())
    def model1(self):
        x3 = self.data[['rpk', 'cell_diagnosis']]
        x3 = x3.groupby('cell_diagnosis').apply(lambda x: xa.merge([
            x.rpk.mean(dim='cell_id').rename('m'),
            (x.rpk**2).mean(dim='cell_id').rename('s')
        ])).todense()
        x3 = (x3.s-x3.m**2)**0.5

        x5 = self.data[['c5ar1', 'cell_diagnosis']].sel(coef='rpk').drop('coef')
        x5 = x5.groupby('cell_diagnosis').apply(lambda x: x.c5ar1.std(dim='cell_id'))

        x2 = self.model.X
        x2 = x2.sel(coef='rpk')*x5/x3
        return x2

    @compose(property, lazy)
    def sigs(self):
        from decoi_atlas.sigs import sigs
        return sigs.all1.rename('s')

    @compose(property, lazy, XArrayCache())
    def t(self):
        import sparse        
        from decoi_atlas.sigs.entrez import symbol_entrez

        x7 = xa.merge([
            self.model.drop('X'),
            self.model1.rename('X')
        ])
        x7 = xa.DataArray(
            np.where(x7.q<0.1, x7.X, 0),
            [x7.cell_diagnosis, x7.feature_id]
        )

        x6 = x7.feature_id.to_series()
        x6 = symbol_entrez(x6)
        x6 = x6.rename(
            symbol='feature_id',
            Entrez_Gene_ID = 'gene'
        )
        x7 = xa.apply_ufunc(
            np.matmul, x6, x7,
            input_core_dims=[['gene', 'feature_id'], ['feature_id', 'cell_diagnosis']],
            output_core_dims=[['gene', 'cell_diagnosis']],
            join='inner'
        )
        x7 = x7/x6.sum(dim='feature_id').todense()
        x7.data = sparse.COO(x7.data)

        x9 = x6.to_series_sparse().reset_index()
        x9 = x9.groupby('gene').feature_id.apply(lambda x: ','.join(x)).rename('symbol')
        x9 = x9.to_xarray().rename('feature_ids')

        x7 = xa.merge([x7.rename('t'), x9], join='inner')

        return x7

    @compose(property, lazy, XArrayCache())
    def gsea(self):
        from decoi_atlas.sigs.fit import fit_gsea
        x8 = xa.merge([self.t, self.sigs], join='inner')
        x8 = fit_gsea(x8.t, x8.s, 1e5)
        return x8

    @compose(property, lazy)
    def result(self):
        x9 = self.gsea
        x9 = x9[['sig','ES', 'NES', 'padj','size']]
        x9 = x9.to_dataframe().reset_index()
        x9 = x9.pivot_table(index='sig', columns='cell_diagnosis', values=['ES', 'NES', 'padj', 'size'])
        x9 = x9[x9['padj'].min(axis=1)<0.05]
        x9 = x9[x9['size'].min(axis=1)>5]
        x9 = x9['NES']
        x9 = x9.add(-x9['control'], axis=0).assign(control=x9['control'])
        x9.columns.name=None
        x9 = x9.reset_index()
        return x9

# %%
class _analysis1(_c5ar1_analysis):
    storage = storage/'analysis1'

    @compose(property, lazy)
    def data(self):
        x2 = super(_analysis1, self).data.copy()
        x2.c5ar1.data[0,:] = np.where(x2.c5ar1.data[0,:]>0, 1, 0)
        return x2

analysis1 = _analysis1()

# %%
class _analysis2(_c5ar1_analysis):
    storage = storage/'analysis2'

    @compose(property, lazy)
    def data(self):
        x2 = super(_analysis2, self).data
        x3 = (x2.c5ar1.sel(coef='rpk')>0).drop('coef')
        x2 = x2.sel(cell_id=x3)
        return x2

analysis2 = _analysis2()

# %%
class _analysis3(_c5ar1_analysis):
    storage = storage/'analysis3'

analysis3 = _analysis3()

# %%
class _analysis4(_c5ar1_analysis):
    storage = storage/'analysis4'

    @compose(property, lazy)
    def data(self):
        x2 = super(_analysis4, self).data
        x2 = x2.sel(
            cell_id=~x2['cell_integrated_snn_res.0.3'].isin([5,7,9])
        )

        return x2

analysis4 = _analysis4()

# %%

# %%
class _spotfire:
    storage = storage/'spotfire'

    @compose(property, lazy, CSVCache(ext='.csv'))
    def result(self):
        return pd.concat([
            analysis1.result.assign(source='analysis1'),
            analysis2.result.assign(source='analysis2')
        ])

spotfire = _spotfire()

# %%
if __name__ == '__main__':
    1 == 1

    # %%
    spotfire.result