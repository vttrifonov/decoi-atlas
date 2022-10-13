# %%
import numpy as np
import pandas as pd
import xarray as xa
import matplotlib.pyplot as plt
from plotnine import *
import statsmodels.api as sm
from decoi_atlas._data import data

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
x3 = sm.stats.Table.from_data(x2[['cell_integrated_snn_res.0.3', 'C5AR1>0']])
print(
    plot_table(x3)
)


# %%
x3 = x2[['cell_integrated_snn_res.0.3', 'rpk_C5AR1']]
x3 = x3[x3.rpk_C5AR1>0].copy()
x3['C5AR1>40'] = np.where(x3.rpk_C5AR1>40, 'C5AR1>40', 'C5AR1<40')
x3 = sm.stats.Table.from_data(x3[['cell_integrated_snn_res.0.3', 'C5AR1>40']])
print(
    plot_table(x3)
)


# %%
print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='cell_integrated_snn_res.0.3'))
)

# %%
print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(x2[x2.counts_C5AR1==0], color='black', alpha=0.1)
        #geom_point(x2[x2.counts_C5AR1>0], color='black', alpha=0.05)
)

# %%
print(
    ggplot(x2[x2.rpk_C5AR1>40])+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(
            #aes(color='rpk_C5AR1>40'), 
            alpha=0.1
        )
)

# %%
plt.hist(x2[x2.rpk_C5AR1>0].rpk_C5AR1, 100)