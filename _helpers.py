#%%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

from pathlib import Path
import pandas as pd
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
