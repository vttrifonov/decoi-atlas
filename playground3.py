# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
from decoi_atlas._data import data

# %%
x = data.c2_wb_pbmc

# %%
x['rpk'] = x.counts.copy()
x.rpk.data = x.rpk.data.asformat('coo')
x['total'] = x.rpk.sum(dim='feature_id').todense()
x['rpk'] = 1e4*x.rpk/x.total

# %%
import statsmodels.api as sm
x2 = x.sel(feature_id=['C5AR1', 'C5AR2'])
x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
x2 = x2.sel(cell_id=~x2['cell_blueprint.labels'].isin(['Macrophages', 'Erythrocytes']))
x2 = x2.todense()
x2 = xa.merge([
    x2.counts.assign_coords(
        feature_id=['counts_'+k for k in x2.feature_id.data]
    ).to_dataset(dim='feature_id'),
    x2.rpk.assign_coords(
        feature_id=['rpk_'+k for k in x2.feature_id.data]
    ).to_dataset(dim='feature_id'),
    x2.drop_dims('feature_id')
])
x2 = x2.to_dataframe()
x2['C5AR1>0'] = np.where(x2.counts_C5AR1>0, 'C5AR1>0', 'C5AR1=0')
x2['C5AR2>0'] = np.where(x2.counts_C5AR2>0, 'C5AR2>0', 'C5AR2=0')
x2['C5AR>0'] = x2['C5AR1>0']+','+x2['C5AR2>0']

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
x3 = sm.stats.Table.from_data(x2[['cell_blueprint.labels', 'cell_diagnosis']])
print(
    plot_table(x3)
)

x3 = sm.stats.Table.from_data(x2[['cell_blueprint.labels', 'C5AR1>0']])
print(
    plot_table(x3)+labs(x='')
)

x3 = sm.stats.Table.from_data(x2[['cell_diagnosis', 'C5AR1>0']])
print(
    plot_table(x3)+labs(x='')
)

x3 = x2[x2['cell_blueprint.labels']=='Neutrophils']
x3 = sm.stats.Table.from_data(x3[['cell_diagnosis', 'C5AR1>0']])
print(
    plot_table(x3)+labs(x='', title='Neutrophils')
)

# %%
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import beta

def ols(A, B):
    Q, R = np.linalg.qr(A)
    nz = ~np.isclose(np.diag(R), 0)
    Q, R = Q[:,nz], R[np.ix_(nz, nz)]
    x = (Q.T) @ B
    SS = (x**2).sum(axis=0)
    x = np.linalg.inv(R) @ x
    X = np.full((A.shape[1], B.shape[1]), np.nan)
    X[nz,:] = x
    return X, SS, Q.shape[1]

def anova(A1, A2, B):
    _, SS1, rk1 = ols(A1, B)
    X, SS2, rk2 = ols(A2, B)
    r2 = (SS2-SS1)/((B**2).sum(axis=0)-SS1)
    df1, df2 = rk2-rk1, B.shape[0]-rk2
    p = beta.sf(r2, df1/2, df2/2)
    return X, r2, p, df1, df2

# %%
x2 = x.sel(feature_id=['C5AR1'])
x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
x2 = x2.sel(cell_id=x2['cell_blueprint.labels']=='Neutrophils')
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

def _(x2):
    x4 = xa.apply_ufunc(
            anova, xa.DataArray(1, [x2.cell_id, ('coef1', [0])]), x2.c5ar1, x2.rpk,
            input_core_dims=[['cell_id', 'coef1'], ['cell_id', 'coef'], ['cell_id', 'feature_id']],
            output_core_dims=[['coef', 'feature_id']]+[['feature_id']]*2+[[]]*2
    )
    x4 = [x.rename(k) for x, k in zip(x4, ['X', 'r2', 'p', 'df1', 'df2'])]
    x4 = xa.merge(x4)
    x5 = x4.p.data
    x5[~np.isnan(x5)] = multipletests(x5[~np.isnan(x5)], method='fdr_bh')[1]
    x4['q'] = 'feature_id', x5
    return x4

x4 = x2.groupby('cell_diagnosis').apply(_)
    
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

x4 = x6.to_series_sparse().reset_index()
x4 = x4.groupby('gene').feature_id.apply(lambda x: ','.join(x)).rename('symbol')
x4 = x4.to_xarray()

# %%
x9 = x8[['sig', 'ES', 'NES', 'padj']]
x9 = x9.to_dataframe().reset_index()
x9 = x9[x9.padj<0.1]
x9 = x9[x9.ES>0]
x9 = x9.sort_values('padj')

x10 = x8.sel(sig='HALLMARK_INTERFERON_GAMMA_RESPONSE')
x10 = x10.sel(cell_diagnosis='COVID-19, severe')
x10 = x10.sel(gene=(x10.leadingEdge==1).todense())
x10 = x7.t.sel(cell_diagnosis=x10.cell_diagnosis.data).\
    sel(gene=x10.gene.data).\
    todense()
x10 = xa.merge([x10, x4], join='inner')
x10 = x10.to_dataframe().reset_index()
# %%
