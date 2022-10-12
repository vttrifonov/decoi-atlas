# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
import matplotlib.pyplot as plt
from decoi_atlas._data import data
import decoi_atlas.common.helpers

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
from scipy.stats import beta
x2 = x.sel(feature_id=['C5AR1'])
x2 = x2.sel(cell_id=x2.cell_diagnosis!='')
x2 = x2.sel(cell_id=x2['cell_blueprint.labels']=='Neutrophils')
x2 = (x2.counts>0).todense()
x2 = x2.squeeze('feature_id').drop('feature_id')
x2 = pd.get_dummies(x2.to_series()).to_xarray().to_array(dim='coef')

x3 = x.rpk.copy()
x3.data = x3.data.asformat('gcxs', compressed_axes=(0,))
x2 = xa.merge([x2.rename('c5ar1'), x3], join='inner')

def ols(A, B):
    Q, R = np.linalg.qr(A)
    nz = ~np.isclose(np.diag(R), 0)
    Q, R = Q[:,nz], R[np.ix_(nz, nz)]
    x = (Q.T) @ B
    RSS = (B**2).sum(axis=0)-(x**2).sum(axis=0)
    x = np.linalg.inv(R) @ x
    X = np.full((A.shape[1], B.shape[1]), np.nan)
    X[nz,:] = x
    return X, RSS, Q.shape[1]

A1, A2, B = np.ones((x2.sizes['cell_id'], 1)), x2.c5ar1.data, x2.rpk.data
_, RSS1, rk1 = ols(A1, B)
X, RSS2, rk2 = ols(A2, B)
r2 = 1-RSS2/RSS1
df1, df2 = B.shape[0]-rk1, B.shape[0]-rk2
p = beta.sf(r2, (df1-df2)/2, df2/2)

z = pd.Series(r2).sort_values(ascending=False)
z[z>0.1]

r2[3345]
z1 = sm.OLS(B[:,[3345]].todense(), A2).fit()
z1.fvalue
X[:,3345]
((df2*r2)/((df1-df2)*(1-r2)))[3345]


x4 = xa.apply_ufunc(
    OLS, x2.c5ar1, x2.rpk,
    input_core_dims=[['cell_id', 'coef'], ['cell_id', 'feature_id']],
    output_core_dims=[['coef', 'feature_id'], ['feature_id']]
)

x4 = x2.c5ar1.data
x4 = np.linalg.qr(x4)
x5 = x2.rpk.data
x5 = (x4[0].T) @ x5
x5 = np.linalg.inv(x4[1]) @ x5





# %%
