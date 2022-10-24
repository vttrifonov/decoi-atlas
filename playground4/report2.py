# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground4'

# %%
from plotnine import *
from matplotlib import colors
import statsmodels.api as sm
import numpy as np
import xarray as xa
import itables
from ..common import helpers
from . import analysis, analysis3, plot_table, quantile

# %%
x2clust = analysis.clust1
x2 = analysis.feature_data1('CD274')

print(
    ggplot()+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(x2[x2.rpk==0], color='gray', alpha=0.05)+
        geom_point(x2[x2.rpk>0], aes(fill='rpk_q3'), shape=1, alpha=1)+
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

x3 = sm.stats.Table.from_data(x2[['cell_integrated_snn_res.0.3', 'rpk_q3']])
print(
    plot_table(x3)
)

x3 = sm.stats.Table.from_data(x2[['cell_diagnosis', 'rpk_q3']])
print(
    plot_table(x3)
)

# %%
x3 = analysis3.data
x3 = x3.sel(feature_id=['CD274']).todense()
x3 = x3.sel(coef='rpk')
x3 = x3.to_dataframe().reset_index()
x3['c5ar1_q3'] = quantile(np.expm1(x3.c5ar1))
x3['cd274_q3'] = quantile(np.expm1(x3.rpk))

x4 = sm.stats.Table.from_data(x3[['c5ar1_q3', 'cd274_q3']])
print(
    plot_table(x4)
)



# %%