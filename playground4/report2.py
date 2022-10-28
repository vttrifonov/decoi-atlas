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
x3 = xa.merge([
    analysis.data.drop_dims(['feature_id', 'umap_dim']),
    analysis3.data
], join='inner')
x3 = x3.sel(feature_id=['CD274']).todense()
x3 = x3.sel(coef='rpk')
x3 = x3.to_dataframe().reset_index()
x3['c5ar1_q3'] = quantile(np.expm1(x3.c5ar1), q=3)
x3['cd274_q3'] = quantile(np.expm1(x3.rpk), q=3)
x3 = x3[x3['cell_integrated_snn_res.0.3']==2]

x4 = x3[x3.cell_diagnosis=='COVID-19, severe']
x4 = x4.merge(
    x4.groupby('cell_diagnosis').\
        apply(lambda x: x[['c5ar1', 'rpk']].\
            corr().iloc[0,1]
        ).\
        rename('R').reset_index()
)
x4['R'] = 'R: ' + x4['R'].round(2).astype(str)
print(
    ggplot(x4)+aes('c5ar1', 'rpk')+
        geom_point(aes(color='cell_diagnosis'))+
        geom_smooth(method='lm')+
        facet_grid('.~cell_diagnosis+R')+
        labs(x='C5AR1', y='CD274')+        
        theme(
            figure_size=(9, 3),
            legend_position='none'
        ) 
)

x4 = x3[x3.cell_diagnosis=='COVID-19, severe']
x4 = sm.stats.Table.from_data(x4[['c5ar1_q3', 'cd274_q3']])
print(
    plot_table(x4)
)

# %%