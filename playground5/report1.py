# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground5'

# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
from . import analysis, gmm_kl
from .._helpers import plot_table

# %%
x = xa.merge([
    analysis.data, 
    analysis.clust2.clust,
    analysis.clust2.means.rename('clust_means'),
    analysis.clust2.enrich.data.rename({
        k: 'sig_'+k 
        for k in analysis.clust2.enrich.data.keys()
    })
], join='inner')

# %%
x1 = x[['sig_coef', 'sig_p']].to_dataframe().reset_index()
x1 = x1.sort_values('sig_p')
x1 = x1[x1.sig_p<1e-4]

# %%
x1 = analysis.clust2.enrich.clust1(10).data.copy()
x1['clust1'] = x1.clust1.astype('str')[x1.clust1]

x2 = pd.DataFrame(dict(
    mu=x1.mean('sig'),
    sigma=x1.std('sig')
), index=x1.clust1)
plt.figure()
x2.plot('mu', 'sigma', kind='scatter')

x2 = (x1-x1.mean('sig'))/x1.std('sig')
x2 = pd.DataFrame(dict(
    mu=x2.mean('clust1'),
    sigma=x2.std('clust1')
), index=x2.sig)
plt.figure()
x2.plot('mu', 'sigma', kind='scatter')

#x1.data = np.apply_along_axis(np.random.permutation, 0, x1.data)
x4 = analysis.clust2.enrich.clust1(10).svd
x4['clust1'] = x4.clust1.astype('str')[x4.clust1]

plt.figure()
x4['s'].query(pc='pc<70').to_series().\
    pipe(lambda x: x**2/(x**2).sum()).\
    pipe(np.log2).\
    plot(style='.')
plt.figure()
x4['v'].sel(pc=0).to_series().sort_values().plot()
plt.figure()
x4['u'].sel(pc=0).to_series().sort_values().plot()

# %%
x5 = analysis.clust2.enrich.clust1(10).gmm
x5['clust'] = x5.clust.astype('str')[x5.clust]

print(
    x5.pred.argmax('clust').to_series().value_counts()
)

x6 = (x5.means@x4.v)*x4.scale + x4['mean']
x6 = x6.rename('means').to_dataframe().reset_index()
print(
    ggplot(x6)+
        aes('clust1', 'clust')+
        geom_tile(aes(fill='means'))+
        scale_fill_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)


x10 = gmm_kl(x5)

x11 = x10.to_dataframe().reset_index()
print(
    ggplot(x11)+
        aes('clust0', 'clust1')+
        geom_tile(aes(fill='kl'))+
        geom_text(aes(label='np.round(kl)'), color='white')
)

print(
    ggplot(x11)+
        aes('np.clip(kl, 0, 50)')+
        geom_freqpoly()
)

x6 = np.linalg.svd(x5.covs.data, compute_uv=False)**2
x6 = x6/x6.sum(axis=1, keepdims=True)
x6 = xa.DataArray(x6, [x5.clust, ('pc', range(x6.shape[1]))], name='ve')
x6 = x6.to_dataframe().reset_index()
print(
    ggplot(x6[x6.pc<10])+
        aes('pc', 'np.log2(ve)')+
        geom_point(aes(group='clust', color='clust'))
)

# %%
x1 = analysis.clust2.enrich.clust1(20)
x3 = x1.svd
x3 = x3.u * x3.s
x3 = x3.sel(pc=x3.pc<2)
x3['pc'] = xa.DataArray(['x', 'y'], [('pc', [0,1])])[x3.pc]
x3 = x3.to_dataset('pc')
x3 = xa.merge([x3, x1.clust])
x3['pred'] = x3.pred.astype(str)
x3 = x3.to_dataframe().reset_index()

x4 = x3[['sig', 'pred']].copy()
x4['sig'] = x4.sig.str.contains('INTERF')
x4 = sm.stats.Table.from_data(x4)
print(plot_table(x4))

x3['pred1'] = x3.pred=='2'
print(
    ggplot(x3)+aes('x', 'y')+
        geom_point(aes(color='pred1'))
)




# %%
x1 = xa.merge([
    x[['pred', 'cell_integrated_snn_res.0.3']], 
    x.umap.to_dataset(dim='umap_dim')
]).to_dataframe()
x1['pred'] = x1.pred.astype('category')
print(
    ggplot(x1)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='pred'), alpha=0.1)+
        theme(legend_position='none')
)

# %%
x1 = x[['cell_integrated_snn_res.0.3', 'pred']].to_dataframe()
for c in x1:
    x1[c] = x1[c].astype('category')
x1 = sm.stats.Table.from_data(x1)
x1 = pd.concat([
    v.stack().rename(k)
    for k, v in [
        ('table', x1.table_orig), 
        ('resid', x1.resid_pearson), 
        ('fit', x1.fittedvalues)
    ]
], axis=1).reset_index()

x2 = x1.sort_values('resid', ascending=False).\
    drop_duplicates('pred').\
    query('resid>15').\
    sort_values('table').\
    rename(columns={'cell_integrated_snn_res.0.3': 'pred1'})
x2 = x2[['pred', 'pred1']]
    
x3 = x[['cell_integrated_snn_res.0.3', 'pred']].to_dataframe()
for c in x3:
    x3[c] = x3[c].astype('category')
x3 = x3.merge(x2)
x3 = x3[['cell_integrated_snn_res.0.3', 'pred1']]
x3 = sm.stats.Table.from_data(x3)
print(
    plot_table(x3)
)

# %%