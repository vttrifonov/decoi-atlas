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
from .common.caching import compose, lazy, XArrayCache, CSVCache
from ._data import data
from ._helpers import config, plot_table, quantile

# %%
def svd(x, k=None, scale=True, chunks=None, dims=None):
    if dims is None:
        dims = x.dims
    if chunks is None:
        chunks = {k: x.sizes[k] for k in dims}
    if k is None:
        k = min([x.sizes[k] for k in dims])

    import dask.array as da
    x = x.chunk(chunks)
    if scale:
        x = xa.Dataset(dict(
            mat=x,
            mean=x.mean(dim=dims[0]),
            scale=x.std(dim=dims[0])
        ))
        x['scale'] = np.sqrt(x.sizes[dims[0]])*x.scale
        x['scale'] = xa.where(x.scale==0, 1, x.scale)
    else:
        x = xa.Dataset(dict(
            mat=x,
            mean=xa.DataArray(0, [x[dims[1]]]),
            scale=xa.DataArray(1, [x[dims[1]]])
        ))
    x['mat'] = (x.mat-x['mean'])/x['scale']
    x = x.transpose(*dims)
    svd = da.linalg.svd_compressed(
        x.mat.data, 
        k=k, n_power_iter=2
    )
    rand_svd = da.linalg.svd_compressed(
        da.apply_along_axis(
            np.random.permutation, 0, x.mat.data, 
            dtype=x.mat.dtype, shape=(svd[0].shape[0],)
        ),
        k=len(svd[1]), n_power_iter=2
    )
    svd = xa.Dataset(
        dict(
            u=((dims[0], 'pc'), svd[0]),
            s=(('pc',), svd[1]),
            v=((dims[1], 'pc'), svd[2].T),
            rand_s=(('pc',), rand_svd[1])
        ),
        {
            dims[0]: x[dims[0]], 
            dims[1]: x[dims[1]], 
            'pc': range(len(svd[1]))
        }
    )
    svd = xa.merge([svd, x[['mean', 'scale']]])
    svd = svd.compute()
    return svd

def gmm(x1, k, dims=None):
    from sklearn.mixture import GaussianMixture
    if dims is None:
        dims = x1.dims
    x1 = x1.transpose(*dims)
    x2 = GaussianMixture(k, verbose=2).fit(x1.data)
    x3 = xa.Dataset()
    x3['means'] = xa.DataArray(x2.means_, [('clust', range(x2.n_components)), x1[dims[1]]])
    x3['covs'] = xa.DataArray(x2.covariances_, [x3.clust, x3[dims[1]], x3[dims[1]].rename({dims[1]: dims[1]+'_1'})])
    x3['pred'] = xa.DataArray(x2.predict_proba(x1.data), [x1[dims[0]], x3.clust])
    return x3

def gmm_kl(x5): 
    x6 = xa.DataArray(np.linalg.inv(x5.covs), x5.covs.coords).rename(clust='clust1')
    x7 = xa.dot(x6, x5.covs.rename(clust='clust0', pc='pc_2'), dims=['pc_1'])
    x7 = x7.transpose('clust0', 'clust1', 'pc', 'pc_2')
    x7 = xa.DataArray(
        np.vectorize(np.diag, signature='(n,n)->(n)')(x7).sum(axis=2),
        [x7.clust0, x7.clust1]
    )
    x8 = x5.means.rename(clust='clust1')-x5.means.rename(clust='clust0')
    x8 = xa.dot(xa.dot(x8, x6, dims=['pc']), x8.rename(pc='pc_1'), dims=['pc_1'])
    x9 = xa.DataArray(np.log(np.linalg.svd(x5.covs, compute_uv=False)).sum(axis=1), [x5.clust])
    x9 = x9.rename(clust='clust1')-x9.rename(clust='clust0')
    x10 = 0.5*(x7+x8-x5.sizes['pc']+x9)
    x10 = 0.5*(x10+x10.rename(clust0='clust1', clust1='clust0'))
    x10 = x10.rename('kl')
    return x10

class _enrichment:
    @compose(property, lazy, XArrayCache())
    def enrich(self):
        from .sigs._sigs import sigs 
        from .sigs.fit import enrichment
        import sparse

        x1 = xa.merge([
            sigs.all1,
            self.feature_entrez.rename(entrez='gene'),
            self.means
        ], join='inner')
        x1['means'] = xa.apply_ufunc(
            np.matmul, x1.feature_entrez, x1.means,
            input_core_dims=[['gene', 'feature_id'], ['feature_id', 'clust']],
            output_core_dims=[['gene', 'clust']]
        )
        x1['means'] = x1.means/x1.feature_entrez.sum(dim='feature_id')
        x1['means'] = x1.means-x1.means.mean('clust')
        x1['means'] = x1.means/x1.means.std('clust')
        x1.means.data = sparse.as_coo(x1.means.data)
        x1 = enrichment(x1.means, x1.set)
        return x1

class _gmm:
    k = 10

    @property
    def svd_k(self):
        return 5*self.k

    @compose(property, lazy, XArrayCache())
    def svd(self):
        print('svd '+str(self.storage))
        svd1 = svd(
            self.data, 
            k = self.svd_k,
            chunks = {'cell_id': 1000, 'feature_id': 1000},
            dims = ['cell_id', 'feature_id']
        )
        return svd1

    @compose(property, lazy, XArrayCache())
    def gmm(self):
        print('gmm '+str(self.storage))
        svd1 = self.svd
        gmm1 = gmm(svd1.u * svd1.s, self.k, dims=['cell_id', 'pc'])
        return gmm1

    @compose(property, lazy, XArrayCache())
    def clust(self):
        if self.k==1:
            return xa.DataArray(
                [0]*self.data.sizes['cell_id'],
                [self.data.cell_id]
            )                    
        return self.gmm.pred.argmax(dim='clust')

    @compose(property, lazy, XArrayCache())
    def means(self):
        if self.k==1:
            x = self.data.mean(dim='cell_id')
            x = x.rename('means')
            x = x.expand_dims(clust=[0])
            return x

        x1 = self.svd
        x2 = self.gmm.means
        x2 = x1.v @ x2
        x2 = x2 * x1.scale + x1['mean']
        x2 = x2.rename('means')
        return x2

def random_data(n, m):
    def _(i):
        u, s, v = np.linalg.svd(np.random.randn(50*2).reshape((50,2)), full_matrices=False)
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
#random_data(2, 2)

# %%
class _analysis:
    storage = config.cache/'playground5'

    @compose(property, lazy)
    def data(self):
        x = data.c2_neutrophils_integrated.copy()
        x1 = x.counts
        x1 = x1.chunk({'cell_id': x.sizes['cell_id']//8, 'feature_id': x.sizes['feature_id']//4})
        x1 = x1.map_blocks(lambda x: x.todense()).persist()
        x['total_cell'] = x1.sum(dim='feature_id').persist()
        x['total_feature'] = x1.sum(dim='cell_id').persist()
        
        x2 = np.log1p(1e4*x1/x.total_cell).persist()
        x['log1p_rpk'] = x2
        x['mean'] = x2.mean(dim='cell_id')
        x['std'] = x2.std(dim='cell_id')

        x = x.compute()
        x = x.sel(feature_id=x['total_feature']>0)

        return x

    @compose(property, lazy, XArrayCache())
    def feature_entrez(self):
        #self = analysis
        from .sigs.entrez import symbol_entrez

        x6 = self.data.feature_id.to_series()
        x6 = symbol_entrez(x6)
        x6 = x6.rename(
            symbol='feature_id',
            Entrez_Gene_ID = 'entrez'
        )
        return x6.rename('feature_entrez')

    @compose(property, lazy)
    def clust1(self):
        class _item:
            @property
            def storage(self):
                return self.prev.storage/'item'/str(self.c)

            @property
            def data(self):
                prev = self.prev
                return prev.data.sel(cell_id=prev.prev.clust==self.c)

            @property
            def k(self):
                prev = self.prev
                k = prev.prev.clust.to_series().value_counts()
                k = np.ceil(prev.k*k/k.sum()).astype(int)
                k = k[self.c]
                return k

        class _split:
            @property
            def storage(self):
                return self.prev.storage/'split'

            @property
            def k(self):
                return 10*self.prev.k

            @property
            def data(self):
                return self.prev.data
            
            @property
            def items(self):
                return list(self.prev.clust.to_series().drop_duplicates())

            def item(self, c):
                item = self._item()
                item.prev = self
                item.c = c
                return item
            
            @property
            def clust(self):
                c = [self.item(c).clust for c in self.items]
                c1 = np.cumsum([c.data.max()+1 for c in c])
                c1 = [0]+list(c1[:-1])
                c = [c+c1 for c, c1 in zip(c, c1)]
                c = xa.concat(c, dim='cell_id')
                return c 

            @property
            def means(self):
                c = [self.item(c).means for c in self.items]
                c1 = np.cumsum([c.clust.data.max()+1 for c in c])
                c1 = [0]+list(c1[:-1])
                c = [
                    c.assign_coords(
                        clust1=('clust', c.clust.data+c1)
                    ).swap_dims(clust='clust1').\
                        drop('clust').rename(clust1='clust')
                    for c, c1 in zip(c, c1)
                ]
                c = xa.concat(c, dim='clust')
                return c

        class _gmm1(_gmm, _enrichment):
            @compose(property, lazy)
            def split(self):
                split = self._split()
                split.prev = self
                return split

            @property
            def feature_entrez(self):
                return self.prev.feature_entrez

        class _item1(_item, _gmm1):
            pass

        class _split1(_split, _enrichment):
            _item = _item1

            @property
            def feature_entrez(self):
                return self.prev.feature_entrez
                
        _gmm1._split = _split1

        class _gmm2(_gmm1):
            storage = self.storage/'clust1'
            prev = self

            @compose(property, lazy)
            def data(self):
                return self.prev.data.log1p_rpk.todense()

        return _gmm2().split

    @compose(property, lazy)
    def clust2(self):
        class _gmm1(_gmm, _enrichment):
            storage = self.storage/'clust2'
            prev = self
            k = 100

            @compose(property, lazy)
            def data(self):
                return self.prev.data.log1p_rpk.todense()

            @property
            def feature_entrez(self):
                return self.prev.feature_entrez

        return _gmm1()

analysis = _analysis()

self = analysis

# %%
x = xa.merge([
    analysis.data, 
    analysis.clust2.clust,
    analysis.clust2.means.rename('clust_means'),
    analysis.clust2.enrich.rename({k: 'sig_'+k for k in analysis.clust1.enrich.keys()})
], join='inner')

# %%
x1 = x[['sig_coef', 'sig_p']].to_dataframe().reset_index()
x1 = x1.sort_values('sig_p')
x1 = x1[x1.sig_p<1e-4]

# %%
x1 = x.sig_coef.fillna(0).transpose('sig', 'clust')
x1['clust'] = x1.clust.astype('str')[x1.clust]

x2 = pd.DataFrame(dict(
    mu=x1.mean('sig'),
    sigma=x1.std('sig')
), index=x1.clust)
plt.figure()
x2.plot('mu', 'sigma', kind='scatter')

x2 = (x1-x1.mean('sig'))/x1.std('sig')
x2 = pd.DataFrame(dict(
    mu=x2.mean('clust'),
    sigma=x2.std('clust')
), index=x2.sig)
plt.figure()
x2.plot('mu', 'sigma', kind='scatter')

#x1.data = np.apply_along_axis(np.random.permutation, 0, x1.data)
x4 = svd(x1, scale=True)

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
x5 = x4#.query(pc='pc<99')
x5 = x5.u * x5.s
#x5.data = np.apply_along_axis(np.random.permutation, 0, x5.data)
x5 = gmm(x5, 20)
x5['clust'] = x5.clust.astype('str')[x5.clust]

print(
    x5.pred.argmax('clust').to_series().value_counts()
)

x6 = (x5.means.rename(clust='sig_clust')@x4.v)*x4.scale + x4['mean']
x6 = x6.rename('means').to_dataframe().reset_index()
print(
    ggplot(x6)+
        aes('clust', 'sig_clust')+
        geom_tile(aes(fill='means'))+
        scale_fill_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)

x6 = xa.DataArray(np.linalg.inv(x5.covs), x5.covs.coords).rename(clust='clust1')
x7 = xa.dot(x6, x5.covs.rename(clust='clust0', pc='pc_2'), dims=['pc_1'])
x7 = x7.transpose('clust0', 'clust1', 'pc', 'pc_2')
x7 = xa.DataArray(
    np.vectorize(np.diag, signature='(n,n)->(n)')(x7).sum(axis=2),
    [x7.clust0, x7.clust1]
)
x8 = x5.means.rename(clust='clust1')-x5.means.rename(clust='clust0')
x8 = xa.dot(xa.dot(x8, x6, dims=['pc']), x8.rename(pc='pc_1'), dims=['pc_1'])
x9 = xa.DataArray(np.log(np.linalg.svd(x5.covs, compute_uv=False)).sum(axis=1), [x5.clust])
x9 = x9.rename(clust='clust1')-x9.rename(clust='clust0')
x10 = 0.5*(x7+x8-x5.sizes['pc']+x9)
x10 = 0.5*(x10+x10.rename(clust0='clust1', clust1='clust0'))
x10 = x10.rename('kl')

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