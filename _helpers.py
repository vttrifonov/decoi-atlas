#%%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

from pathlib import Path
import pandas as pd
import numpy as np
from plotnine import *

#%%

class _config:
    project='decoi-atlas'
    cache = Path.home()/'.cache'/project
    root = Path(__file__).parent

config = _config()

# %%
def plot_table(x3, show_obs=True, show_exp=True):
    x1 = [x3.table_orig.index.name, x3.table_orig.columns.name]
    x3 = pd.concat([
        v.stack().rename(k)
        for k, v in [
            ('table', x3.table_orig), 
            ('resid', x3.resid_pearson), 
            ('fit', x3.fittedvalues)
        ]
    ], axis=1).reset_index()
    if show_exp:
        x3['delta'] = x3['table'].astype(str) + '\n' + x3['fit'].astype(int).astype(str)
    elif show_obs:
        x3['delta'] = x3['table'].astype(str)
    else:
        x3['delta'] = ['']*x3.shape[0]

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
def quantile(x, q=3):
    x1 = np.where(x==0, np.nan, x)
    x1 = pd.qcut(x1, q=q)
    x1 = x1.add_categories(pd.Interval(0,0))
    x1 = x1.reorder_categories(np.roll(np.array(x1.dtype.categories), 1))
    x1 = x1.fillna(pd.Interval(0,0))
    return x1