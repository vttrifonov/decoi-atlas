# %%
#import os
#os.environ['OMP_THREAD_LIMIT']='1'
#os.environ['MKL_THREAD_LIMIT']='1'

# %%
import numpy as np
import pandas as pd
import xarray as xa
import matplotlib.pyplot as plt
from matplotlib import colors
from plotnine import *
import statsmodels.api as sm
import decoi_atlas.common.helpers
from decoi_atlas.common.caching import compose, lazy, XArrayCache
from decoi_atlas._data import data
from decoi_atlas._helpers import config

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
x = data.c2_neutrophils_integrated

# %%
x['rpk'] = x.counts.copy()
x.rpk.data = x.rpk.data.asformat('coo')
x['total'] = x.rpk.sum(dim='feature_id').todense()
x['rpk'] = 1e4*x.rpk/x.total

# %%
x2 = x.sel(feature_id=['C5AR1', 'C5AR2'])
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
x2['C5AR2>0'] = np.where(x2.counts_C5AR2>0, 'C5AR2>0', 'C5AR2=0')
x2['C5AR>0'] = x2['C5AR1>0']+','+x2['C5AR2>0']
x2['cell_integrated_snn_res.0.3'] = x2['cell_integrated_snn_res.0.3'].astype('category')

x2clust = x2.groupby('cell_integrated_snn_res.0.3')[['UMAP_1', 'UMAP_2']].\
    mean().reset_index()

def plot_table(x3):
    x1 = [x3.table_orig.index.name, x3.table_orig.columns.name]
    x3 = pd.concat([
        x3.table_orig.reset_index().\
            melt(id_vars=x1[:1]).\
            rename(columns={'value': 'table'}).\
            set_index(x1),
        x3.resid_pearson.reset_index().\
            melt(id_vars=x1[:1]).\
            rename(columns={'value': 'resid'}).\
            set_index(x1),
        x3.fittedvalues.reset_index().\
            melt(id_vars=x1[:1]).\
            rename(columns={'value': 'fit'}).\
            set_index(x1)
    ], axis=1).reset_index()
    x3['delta'] = x3['table'].astype(str) + '\n' + x3['fit'].astype(int).astype(str)

    return (
        ggplot(x3)+
            aes(x1[1], x1[0])+
            geom_tile(aes(fill='resid'))+
            geom_text(aes(label='delta'))+
            scale_fill_gradient2(
                low='blue', mid='white', high='red',
                midpoint=0
            )
    )

# %%
print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='cell_integrated_snn_res.0.3'))+
        geom_label(
            x2clust, 
            aes(
                label='cell_integrated_snn_res.0.3',
                color='cell_integrated_snn_res.0.3',
            ),
            fill=colors.to_hex(colors.to_rgb('white')+(0.8,), keep_alpha=True), 
            size=15
        )+
        theme(legend_position='none')
)

print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(x2[x2.counts_C5AR1==0], color='black', alpha=0.05)+
        geom_label(
            x2clust, 
            aes(
                label='cell_integrated_snn_res.0.3',
                color='cell_integrated_snn_res.0.3',
            ),
            fill=colors.to_hex(colors.to_rgb('white')+(0.8,), keep_alpha=True), 
            size=15
        )+
        theme(legend_position='none')
)

print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(x2[x2.counts_C5AR1>0], color='black', alpha=0.05)+
        geom_label(
            x2clust, 
            aes(
                label='cell_integrated_snn_res.0.3',
                color='cell_integrated_snn_res.0.3',
            ),
            fill=colors.to_hex(colors.to_rgb('white')+(0.8,), keep_alpha=True), 
            size=15
        )+
        theme(legend_position='none')
)

x3 = sm.stats.Table.from_data(x2[['cell_integrated_snn_res.0.3', 'C5AR1>0']])
print(
    plot_table(x3)
)

x3 = sm.stats.Table.from_data(x2[['cell_diagnosis', 'C5AR1>0']])
print(
    plot_table(x3)
)


# %%
x3 = x2[x2.rpk_C5AR1>0].copy()
x3['rpk_C5AR1_q4'] = pd.qcut(x3.rpk_C5AR1, q=3)
print(
    ggplot(x3)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(
            aes(color='rpk_C5AR1_q4'), 
            alpha=0.5
        )
)

x4 = sm.stats.Table.from_data(x3[['cell_integrated_snn_res.0.3', 'rpk_C5AR1_q4']])
print(
    plot_table(x4)
)

x4 = sm.stats.Table.from_data(x3[['cell_diagnosis', 'rpk_C5AR1_q4']])
print(
    plot_table(x4)
)

# %%
plt.hist(np.clip(x2.rpk_C5AR1, 0, 80), 200)

# %%
class _analysis1:
    storage = config.cache/'playground4'/'analysis1'

    @compose(property, lazy)
    def data(self):
        x2 = x.sel(feature_id=['C5AR1'])
        x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
        x2 = xa.merge([
            (x2.counts>0).todense(),
            x2.cell_diagnosis
        ])
        x2 = x2.squeeze('feature_id').drop('feature_id')
        x2 = xa.merge([
            x2.cell_diagnosis,
            pd.get_dummies(x2.counts.to_series()).to_xarray().\
                to_array(dim='coef').rename('c5ar1')
        ])
        x2.c5ar1[0,:] = 1

        x3 = x.rpk.copy()
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
    def gsea(self):
        import sparse
        from decoi_atlas.sigs import sigs
        from decoi_atlas.sigs.fit import fit_gsea
        from decoi_atlas.sigs.entrez import symbol_entrez

        x4 = self.model

        x6 = x4.feature_id.to_series()
        x6 = symbol_entrez(x6)
        x6 = x6.rename(
            symbol='feature_id',
            Entrez_Gene_ID = 'gene'
        )
        x7 = xa.DataArray(
            np.where(x4.q<0.1, x4.X.sel(coef=True), 0),
            [x4.cell_diagnosis, x4.feature_id]
        )
        x7 = xa.apply_ufunc(
            np.matmul, x6, x7,
            input_core_dims=[['gene', 'feature_id'], ['feature_id', 'cell_diagnosis']],
            output_core_dims=[['gene', 'cell_diagnosis']],
            join='inner'
        )
        x7 = x7/x6.sum(dim='feature_id').todense()
        x7 = xa.merge([x7.rename('t'), sigs.all1.rename('s')], join='inner')
        x7.t.data = sparse.COO(x7.t.data)
        x8 = fit_gsea(x7.t, x7.s, 1e5)

        x9 = x6.to_series_sparse().reset_index()
        x9 = x9.groupby('gene').feature_id.apply(lambda x: ','.join(x)).rename('symbol')
        x9 = x9.to_xarray().rename('feature_ids')

        x8 = xa.merge([x8, x7.t, x6, x9], join='inner')

        return x8

analysis1 = _analysis1()

# %%
analysis1.data

analysis1.model

analysis1.gsea


# %%
import sparse
from decoi_atlas.sigs import sigs
from decoi_atlas.sigs.fit import fit_gsea
from decoi_atlas.sigs.entrez import symbol_entrez

x4['t'] = ('cell_diagnosis', 'feature_id'), np.where(x4.q<0.1, x4.X.sel(coef=True), 0)

x6 = x4.feature_id.to_series()
x6 = symbol_entrez(x6)
x6 = x6.rename(
    symbol='feature_id',
    Entrez_Gene_ID = 'gene'
)
x7 = xa.apply_ufunc(
    np.matmul, x6, x4.t,
    input_core_dims=[['gene', 'feature_id'], ['feature_id', 'cell_diagnosis']],
    output_core_dims=[['gene', 'cell_diagnosis']],
    join='inner'
)
x7 = x7/x6.sum(dim='feature_id').todense()
x7 = xa.merge([x7.rename('t'), sigs.all1.rename('s')], join='inner')
x7.t.data = sparse.COO(x7.t.data)
x8 = fit_gsea(x7.t, x7.s, 1e5)

x6 = x6.to_series_sparse().reset_index()
x6 = x6.groupby('gene').feature_id.apply(lambda x: ','.join(x)).rename('symbol')
x6 = x6.to_xarray()

# %%
x8 = analysis1.gsea

x9 = x8[['sig', 'ES', 'NES', 'padj']]
x9 = x9.to_dataframe().reset_index()
x9 = x9[x9.padj<0.1]
x9 = x9[x9.ES>0]
x9 = x9.sort_values('padj')

# %%
x10 = x8.sel(sig='HALLMARK_INTERFERON_GAMMA_RESPONSE')
x10 = x10.sel(cell_diagnosis='COVID-19, severe')
x10 = x10.sel(gene=(x10.leadingEdge==1).todense())
x10 = x10.drop_dims('feature_id')
x10 = x10[['t', 'feature_ids']].todense()
x10 = x10.to_dataframe().reset_index()
# %%
