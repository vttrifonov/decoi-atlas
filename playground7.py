# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
import matplotlib.pyplot as plt
import statsmodels.api as sm

from .common import helpers
from .common.caching import compose, lazy, XArrayCache
from ._data import data
from ._helpers import config, plot_table, quantile

# %%
x1 = data.c1_pbmc
x1 = x1.rename(cell_group_per_sample='cell_diagnosis')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis']]
x1 = x1.to_dataframe()
x1 = sm.stats.Table.from_data(x1)
print(plot_table(x1))

# %%
x1 = data.c1_pbmc
x1 = x1.rename(cell_group_per_sample='cell_diagnosis')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']]
x1 = x1.to_dataframe()
x2 = x1.groupby(['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']).size()
x3 = x1.groupby(['cell_diagnosis', 'cell_donor']).size()
x4 = pd.merge(
    x2.rename('n').reset_index(), 
    x3.rename('m').reset_index()
)
x4['freq'] = x4.n/x4.m
x4 = x4[x4['cell_blueprint.labels']=='NK cells']
x4 = x4[x4['cell_diagnosis']!='']
print(
    ggplot(x4)+aes('cell_diagnosis', '100*freq')+
        geom_violin(aes(fill='cell_diagnosis'))+
        geom_boxplot(aes(fill='cell_diagnosis'), width=0.2)+
        geom_jitter(width=0.1)+
        labs(x='', y='NK cells frequency (%)')+
        theme(
            figure_size=(4, 3),
            legend_position='none'
        ) 
)

# %%
x1 = data.c2_wb_pbmc
x1 = x1.sel(cell_id=x1.cell_purification=='FreshPBMC')
x1 = x1.sel(cell_id=x1.cell_diagnosis!='')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis']]
x1 = x1.to_dataframe()
x1 = sm.stats.Table.from_data(x1)
print(plot_table(x1))

# %%
x1 = data.c2_wb_pbmc
x1 = x1.sel(cell_id=x1.cell_purification=='FreshPBMC')
x1 = x1.sel(cell_id=x1.cell_diagnosis!='')
x1 = x1[['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']]
x1 = x1.to_dataframe()
x2 = x1.groupby(['cell_blueprint.labels', 'cell_diagnosis', 'cell_donor']).size()
x3 = x1.groupby(['cell_diagnosis', 'cell_donor']).size()
x4 = pd.merge(
    x2.rename('n').reset_index(), 
    x3.rename('m').reset_index()
)
x4['freq'] = x4.n/x4.m
x4 = x4[x4['cell_blueprint.labels']=='NK cells']
x4 = x4[x4['cell_diagnosis']!='']
print(
    ggplot(x4)+aes('cell_diagnosis', '100*freq')+
        geom_violin(aes(fill='cell_diagnosis'))+
        geom_boxplot(aes(fill='cell_diagnosis'), width=0.2)+
        geom_jitter(width=0.1)+
        labs(x='', y='NK cells frequency (%)')+
        theme(
            figure_size=(5, 4),
            legend_position='none'
        )
)

# %%
