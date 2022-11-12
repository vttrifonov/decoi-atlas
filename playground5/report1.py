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
def random_data(n, m, k):
    from . import gmm, svd
    def _(i):
        u, s, v = np.linalg.svd(np.random.randn(k*2).reshape((k,2)), full_matrices=False)
        s = np.sort(10*np.random.random(2))[::-1]
        x1 = (u*s.reshape((1,-1)))@v+5*np.random.random(2).reshape((1,-1))
        return x1, [str(i)]*x1.shape[0]

    x4 = [_(i) for i in range(n)]
    x4 = [np.concatenate(x, axis=0) for x in zip(*x4)]
    x3 = xa.Dataset(dict(
        pt=xa.DataArray(
            x4[0], 
            [('sample', range(x4[0].shape[0])), ('feature', ['x', 'y'])]
        )
    ))
    x3['clust1'] = 'sample', x4[1]

    x4 = svd(x3.pt, scale=False)

    x5 = x4.query(pc='pc>=0')
    x5 = x5.u*x5.s
    x5 = gmm(x5, m)
    x5['clust'] = x5.clust.astype(str)[x5.clust]

    x6 = x5.covs.groupby('clust').\
        apply(
            lambda x: 
                svd(x.rename(pc='pc_2'), scale=False)[['u', 's']].\
                    rename(pc='pc1')
        ).rename(pc_2='pc')
    x6 = 2*x6.u*np.sqrt(x6.s) + x5.means
    x6 = (x6 @ x4.v)*x4.scale + x4['mean']
    x7 = (x5.means @ x4.v)*x4.scale + x4['mean']

    x6 = x6.to_dataset(dim='feature')
    x6 = x6.to_dataframe().reset_index()

    x7 = x7.to_dataset(dim='feature')
    x7 = x7.to_dataframe().reset_index()
    x7['pc1'] = -1

    x8 = xa.merge([
        x3.pt.to_dataset(dim='feature'),
        x3.drop_dims('feature')
    ]).to_dataframe().reset_index()
    x9 = pd.concat([x6, x7])

    x10 = gmm_kl(x5).to_dataframe().reset_index()

    print(
        ggplot()+aes('x', 'y')+
            geom_point(x8, aes(fill='clust1'), shape=0)+
            geom_line(x9[x9.pc1!=0], aes(color='clust'), size=2)+
            geom_line(x9[x9.pc1!=1], aes(color='clust'), size=2)
    )

    print(
        ggplot(x10)+
            aes('clust0', 'clust1')+
            geom_tile(aes(fill='kl'))+
            geom_text(aes(label='np.round(kl)'), color='white')
    )

# %%
#random_data(1, 4, 1000)

# %%
x1 = analysis.clust3(15).clust.to_dataframe()
x1.value_counts()

# %%
x2 = 'cell_integrated_snn_res.0.3'
x1 = analysis.clust3(20)
x1 = xa.merge([analysis.data, x1.clust], join='inner')
x1 = x1[[x2, 'pred']].to_dataframe()
x1 = sm.stats.Table.from_data(x1)
x1 = pd.concat([
    v.stack().rename(k)
    for k, v in [
        ('table', x1.table_orig), 
        ('resid', x1.resid_pearson), 
        ('fit', x1.fittedvalues)
    ]
], axis=1).reset_index()

x3 = x1.sort_values('resid', ascending=False).\
    drop_duplicates('pred').\
    sort_values([x2, 'resid'])

x4 = list(x3[x2].drop_duplicates())
x4 = x4+list(set(x1[x2])-set(x4))
x1[x2] = x1[x2].astype(pd.CategoricalDtype(x4, ordered=True))
x1['pred'] = x1['pred'].astype(pd.CategoricalDtype(x3['pred'].drop_duplicates(), ordered=True))
x1['label'] = x1['table'].astype(str) + '\n' + x1['fit'].astype(int).astype(str)
(
    ggplot(x1)+
        aes(x2, 'pred')+
        geom_tile(aes(fill='resid'))+
        geom_text(aes(label='label'), size=7)+
        scale_fill_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )+
        theme(figure_size=(5, 6))
)

# %%
x1 = analysis.clust3(20)
x1 = xa.merge([analysis.data, x1.clust], join='inner')

x1 = xa.merge([
    x1[['pred', 'cell_integrated_snn_res.0.3']], 
    x1.umap.to_dataset(dim='umap_dim')
]).to_dataframe()
x1['pred'] = x1.pred.astype('category')

print(
    ggplot(x1)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='pred'), alpha=0.1)+
        theme(legend_position='none')
)


# %%
x1 = analysis.clust3(20).enrich.clust1(20).data.copy()
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
x4 = analysis.clust3(20).enrich.clust1(20).svd
x4['clust1'] = x4.clust1.astype('str')[x4.clust1]

plt.figure()
x4['s'].query(pc='pc<70').to_series().\
    pipe(lambda x: x**2/(x**2).sum()).\
    pipe(np.log2).\
    plot(style='.')
plt.figure()
x4['v'].sel(pc=0).to_series().sort_values().plßot()
plt.figure()
x4['u'].sel(pc=0).to_series().sort_values().plot()

# %%
x5 = analysis.clust3(20).enrich.clust1(30).gmm
x5['clust'] = x5.clust.astype('str')[x5.clust]

print(
    x5.pred.argmax('clust').to_series().value_counts()
)

x4 = analysis.clust3(20).enrich.clust1(30).svd
x4['clust1'] = x4.clust1.astype('str')[x4.clust1]
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


x10 = analysis.clust3(20).enrich.clust1(30).kl

x11 = x10.to_dataframe().reset_index()
print(
    ggplot(x11)+
        aes('clust0', 'clust1')+
        geom_tile(aes(fill='kl'))+
        geom_text(aes(label='np.round(kl)', size=7), color='white')+
        theme(figure_size=(8, 8))
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
x1 = analysis.clust3(20).enrich.clust1(30)
x3 = x1.svd
x3 = x3.u * x3.s
x3 = x3.sel(pc=x3.pc<2)
x3['pc'] = xa.DataArray(['x', 'y'], [('pc', [0,1])])[x3.pc]
x3 = x3.to_dataset('pc')
x3 = xa.merge([x3, x1.clust])
x4 = x1.gmm.pred.data
x3['proba'] = 'sig', x4[np.arange(x4.shape[0]), x3.pred.data]
x3['pred'] = x3.pred.astype(str)
x3 = x3.to_dataframe().reset_index()

# %%

x4 = x3[['sig', 'pred']].copy()
x4['sig'] = x4.sig.str.contains('INTERF')
x4 = sm.stats.Table.from_data(x4)
print(plot_table(x4))

# %%
x3['pred1'] = x3.pred=='3'
x3['pred1'] = x3.pred
print(
    ggplot(x3[x3.proba>0.9].sort_values('pred1'))+aes('x', 'y')+
        geom_point(aes(color='pred1'))
)

# %%
print(
    ggplot(x3)+
        aes('proba')+
        geom_freqpoly(bins=100)
)
