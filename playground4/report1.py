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
from . import analysis, analysis1, analysis2, analysis3, plot_table

# %% 
x2, x2clust = analysis.data3
print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='cell_blueprint.labels'))+
        geom_label(
            x2clust, 
            aes(
                label='cell_blueprint.labels',
                color='cell_blueprint.labels',
            ),
            fill=colors.to_hex(colors.to_rgb('white')+(0.8,), keep_alpha=True), 
            size=15
        )+
        theme(legend_position='none')
)

print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(fill='rpk_C5AR1_q3'), shape=1, alpha=0.5)+
        geom_label(
            x2clust, 
            aes(
                label='cell_blueprint.labels',
                color='cell_blueprint.labels',
            ),
            fill=colors.to_hex(colors.to_rgb('white')+(0.8,), keep_alpha=True), 
            size=15
        )+
        theme(legend_position='none')
)

x3 = sm.stats.Table.from_data(x2[['cell_blueprint.labels', 'rpk_C5AR1_q3']])
print(
    plot_table(x3)
)

x3 = sm.stats.Table.from_data(x2[['cell_diagnosis', 'rpk_C5AR1_q3']])
print(
    plot_table(x3)
)

# %%
x2, x2clust = analysis.data1
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
        aes('np.clip(rpk_C5AR1, 0, 80)')+
        geom_freqpoly(bins=200)
)

print(
    ggplot(x2)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(fill='rpk_C5AR1_q3'), shape=1, alpha=0.5)+
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

x3 = sm.stats.Table.from_data(x2[['cell_integrated_snn_res.0.3', 'rpk_C5AR1_q3']])
print(
    plot_table(x3)
)

x3 = sm.stats.Table.from_data(x2[['cell_diagnosis', 'rpk_C5AR1_q3']])
print(
    plot_table(x3)
)

# %%
x3 = analysis3.data
x3 = x3.sel(feature_id=['IFITM2']).todense()
x3 = x3.sel(coef='rpk')
x3 = x3.to_dataframe().reset_index()
x3['c5ar1_nz'] = np.where(x3.c5ar1>0, 'C5AR1>0', 'C5AR1=0')
x3['c5ar1_nz'] = x3['c5ar1_nz'].astype('category')

x4 = x3.copy()
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
        labs(x='C5AR1', y='IFITM2')+        
        theme(
            figure_size=(9, 3),
            legend_position='none'
        ) 
)

x4 = x3.copy()
x4['c5ar1_nz1'] = np.where(x4.c5ar1>0, 1, 0)
x4 = x4.merge(
    x4.groupby('cell_diagnosis').\
        apply(lambda x: x[['c5ar1_nz1', 'rpk']].\
            corr().iloc[0,1]
        ).rename('R').reset_index()
)
x4['R'] = 'R: ' + x4['R'].round(2).astype(str)
print(
    ggplot(x4)+aes('c5ar1_nz', 'rpk')+
        geom_boxplot(aes(color='cell_diagnosis'))+
        facet_grid('.~cell_diagnosis+R')+
        labs(x='C5AR1>0', y='IFITM2')+
        theme(
            figure_size=(9, 3),
            legend_position='none'
        ) 
)

x4 = x3.copy()
x4 = x4[x4.c5ar1>0]
x4 = x4.merge(
    x4.groupby('cell_diagnosis').\
        apply(lambda x: x[['c5ar1', 'rpk']].\
            corr().iloc[0,1]
        ).rename('R').reset_index()
)
x4['R'] = 'R: ' + x4['R'].round(2).astype(str)
print(
    ggplot(x4)+aes('c5ar1', 'rpk')+
        geom_point(aes(color='cell_diagnosis'))+
        geom_smooth(method='lm')+
        facet_grid('.~cell_diagnosis+R')+
        labs(x='C5AR1', y='IFITM2')+
        theme(
            figure_size=(9, 3),
            legend_position='none'
        ) 
)

# %%
x3 = xa.merge([
    analysis1.sigs.sel(sig='HALLMARK_INTERFERON_GAMMA_RESPONSE'),
    analysis1.gsea.sel(sig='HALLMARK_INTERFERON_GAMMA_RESPONSE'),
    analysis1.t
], join='inner').todense()
x3 = x3.to_dataframe().reset_index()
x3['s'] = np.where(x3.s==1, 'yes', 'no')
x3['leadingEdge'] = np.where(x3.leadingEdge==1, 'yes', 'no')
x3['NES'] = 'NES: ' + x3.NES.round(2).astype(str)

print(
    ggplot(x3)+
        aes('s', 'np.clip(t, -0.5, 0.5)')+
        geom_boxplot()+
        geom_point(aes(color='leadingEdge'))+
        geom_label(
            x3[x3.feature_ids=='IFITM2'], 
            aes(label='feature_ids', color='leadingEdge')
        )+
        facet_grid('.~cell_diagnosis+NES')+
        labs(
            x='HALLMARK_INTERFERON_GAMMA_RESPONSE',
            y='t'
        )+
        theme(
            figure_size=(10, 3)
        )
)

# %%
x9 = analysis1.gsea
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
            x9[x9.sig=='HALLMARK_INTERFERON_GAMMA_RESPONSE'],
            aes(label='sig')
        )+
        scale_color_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)

# %%
itables.show(x9, scrollY="400px", scrollCollapse=True, paging=False)

# %%
x3 = xa.merge([
    analysis2.sigs.sel(sig='HALLMARK_INTERFERON_GAMMA_RESPONSE'),
    analysis2.gsea.sel(sig='HALLMARK_INTERFERON_GAMMA_RESPONSE'),
    analysis2.t
], join='inner').todense()
x3 = x3.to_dataframe().reset_index()
x3['s'] = np.where(x3.s==1, 'yes', 'no')
x3['leadingEdge'] = np.where(x3.leadingEdge==1, 'yes', 'no')
x3['NES'] = 'NES: ' + x3.NES.round(2).astype(str)

print(
    ggplot(x3)+
        aes('s', 'np.clip(t, -0.5, 0.5)')+
        geom_boxplot()+
        geom_point(aes(color='leadingEdge'))+
        geom_label(
            x3[x3.feature_ids=='IFITM2'], 
            aes(label='feature_ids', color='leadingEdge')
        )+
        facet_grid('.~cell_diagnosis+NES')+
        labs(
            x='HALLMARK_INTERFERON_GAMMA_RESPONSE',
            y='t'
        )+
        theme(
            figure_size=(10, 3)
        )
)


# %%
x9 = analysis2.gsea
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
            x9[x9.sig=='HALLMARK_INTERFERON_GAMMA_RESPONSE'],
            aes(label='sig')
        )+
        scale_color_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)

# %%
itables.show(x9, scrollY="400px", scrollCollapse=True, paging=False)

# %%