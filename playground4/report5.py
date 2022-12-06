# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground4'

# %%
from plotnine import *
from matplotlib import colors
import statsmodels.api as sm
import numpy as np
import xarray as xa
import pandas as pd
import ipywidgets as ipw
from ..common import helpers
from . import analysis, analysis3, plot_table, quantile
from ..reactive import reactive, observe, VBox, HBox, display

# %%
x2clust = analysis.clust1

# %%
x3 = [''] + list(analysis.data.feature_id.data)

ui1 = HBox(dict(
    regex = ipw.Text(description='feature regex'),    
    feature = ipw.Dropdown(value='', options=x3)
))

@observe(ui1.regex)
def feature_update(regex):
    if regex=='':
        return x3
    x4 = pd.Series(x3)
    x4 = x4[x4.str.contains(regex, regex=True)].to_list()
    ui1.feature.options = [''] + x4

@reactive(ui1.feature)
def x2(feature_id):
    if feature_id=='':
        return None
    return analysis.feature_data1(feature_id)


# %%
ui2 = VBox(dict(
    plot1 = ipw.Output(),
    plot2 = ipw.Output(),
    plot3 = ipw.Output()
))

@reactive(x2)
def plot1(x2):
    if x2 is None:
        return None
    return (
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

@reactive(x2)
def plot2(x2):
    if x2 is None:
        return None
    x3 = sm.stats.Table.from_data(x2[['cell_integrated_snn_res.0.3', 'rpk_q3']])
    return plot_table(x3)

@reactive(x2)
def plot3(x2):
    if x2 is None:
        return None
    x3 = sm.stats.Table.from_data(x2[['cell_diagnosis', 'rpk_q3']])
    return plot_table(x3)

display(ui2.plot1, plot1)
display(ui2.plot2, plot2)
display(ui2.plot3, plot3)

# %%
ipw.VBox([ui1, ui2])

# %%