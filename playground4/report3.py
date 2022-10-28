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
from . import analysis, analysis3, analysis4, plot_table, quantile

# %%
x2clust = analysis.clust1
x2 = analysis.feature_data1('IFITM2')

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
x3 = x3.sel(feature_id=['IFITM2']).todense()
x3 = x3.sel(coef='rpk')
x3 = x3.to_dataframe().reset_index()
x3['c5ar1_q3'] = quantile(np.expm1(x3.c5ar1), q=3)
x3['ifitm2_q3'] = quantile(np.expm1(x3.rpk), q=3)
x3['q3'] = x3['c5ar1_q3'].astype(str) + ':' + x3['ifitm2_q3'].astype(str)
x3['cell_integrated_snn_res.0.3'] = x3['cell_integrated_snn_res.0.3'].astype('category')

x4 = sm.stats.Table.from_data(x3[['cell_integrated_snn_res.0.3', 'q3']])
print(
    plot_table(x4)+
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 12)
        )
)

# %%
x3 = xa.merge([
    analysis.data.drop_dims(['feature_id', 'umap_dim']),
    analysis3.data
], join='inner')
x3 = x3.sel(feature_id=['IFITM2']).todense()
x3 = x3.sel(coef='rpk')
x3 = x3.to_dataframe().reset_index()
x3['c5ar1_q3'] = quantile(np.expm1(x3.c5ar1), q=3)
x3['ifitm2_q3'] = quantile(np.expm1(x3.rpk), q=3)
x1 = x3['cell_integrated_snn_res.0.3']
x1 = np.where(x1.isin([5,7,9]), x1, 'other')
x3['cell_integrated_snn_res.0.3'] = x1

print(
    ggplot(x3)+aes('c5ar1', 'rpk')+
        geom_point(aes(color='cell_integrated_snn_res.0.3'))+
        geom_smooth(method='lm')+
        geom_smooth(
            data=x3[x3['cell_integrated_snn_res.0.3']=='other'],
            method='lm',
            linetype='dashed'
        )+
        facet_grid('.~cell_diagnosis')+
        labs(x='C5AR1', y='IFITM2')+        
        theme(
            figure_size=(9, 3)
        ) 
)

# %%
x3 = xa.merge([
    analysis.data.drop_dims(['feature_id', 'umap_dim']),
    analysis3.data
], join='inner')
x3 = x3.sel(feature_id=['IFITM2']).todense()
x3 = x3.sel(coef='rpk')
x3 = x3.to_dataframe().reset_index()
x3['c5ar1_q3'] = quantile(np.expm1(x3.c5ar1), q=3)
x3['ifitm2_q3'] = quantile(np.expm1(x3.rpk), q=3)
x3 = x3[x3['cell_integrated_snn_res.0.3']==2]

x4 = sm.stats.Table.from_data(x3[['c5ar1_q3', 'ifitm2_q3']])
print(
    plot_table(x4)
)

# %%
x9 = analysis4.gsea
x9 = x9[['sig','ES', 'NES', 'padj','size']]
x9 = x9.to_dataframe().reset_index()
x9 = x9.pivot_table(index='sig', columns='cell_diagnosis', values=['ES', 'NES', 'padj', 'size'])
x9 = x9[x9['padj'].min(axis=1)<0.05]
x9 = x9[x9['size'].min(axis=1)>5]
x9 = x9['NES']
x9 = x9.add(-x9['control'], axis=0).assign(control=x9['control'])
x9.columns.name=None
x9 = x9.reset_index()

# %%
print(
    ggplot(x9)+
        aes('COVID-19, severe', 'COVID-19, mild', color='control')+
        geom_point()+
        geom_label(
            x9[x9.control<0],
            aes(label='sig'),
            adjust_text = {
                'expand_points': (2, 2),
                'arrowprops': {
                    'arrowstyle': '->',
                    'color': 'black'
                }
            }
        )+
        scale_color_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)

# %%
x4 = 'HALLMARK_INTERFERON_GAMMA_RESPONSE'
x3 = xa.merge([
    analysis4.sigs.sel(sig=x4),
    analysis4.gsea.sel(sig=x4),
    analysis4.t
], join='inner').todense()
x3 = x3.to_dataframe().reset_index()
x3['s'] = np.where(x3.s==1, 'yes', 'no')
x3['leadingEdge'] = np.where(x3.leadingEdge==1, 'yes', 'no')
x3['NES'] = 'NES: ' + x3.NES.round(2).astype(str)
x3['ES'] = 'ES: ' + x3.ES.round(2).astype(str)

print(
    ggplot(x3)+
        aes('s', 'np.clip(t, -0.5, 0.5)')+
        geom_boxplot()+
        geom_point(aes(color='leadingEdge'))+
        geom_text(
            x3[x3.feature_ids=='IFITM2'], 
            aes(label='feature_ids', color='leadingEdge')
        )+
        facet_grid('.~cell_diagnosis+NES+ES')+
        labs(
            x=x4,
            y='t'
        )+
        theme(
            figure_size=(10, 3)
        )
)


# %%
itables.show(x9, scrollY="400px", scrollCollapse=True, paging=False)

# %%
