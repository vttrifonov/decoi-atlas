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
        import dask.array as da
        from sklearn.mixture import GaussianMixture

        def svd(x, chunks, k):
            x = x.chunk(chunks)
            x = xa.merge([
                x.rename('mat'),
                x.mean(dim='cell_id').rename('mean'),
                x.std(dim='cell_id').rename('scale')
            ])
            x['scale'] = np.sqrt(x.sizes['cell_id'])*x.scale
            x['scale'] = xa.where(x.scale==0, 1, x.scale)
            x['mat'] = (x.mat-x['mean'])/x['scale']
            x = x.transpose('cell_id', 'feature_id')
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
                    u=(('cell_id', 'pc'), svd[0]),
                    s=(('pc',), svd[1]),
                    v=(('feature_id', 'pc'), svd[2].T),
                    rand_s=(('pc',), rand_svd[1])
                ),
                dict(
                    cell_id = x.cell_id, 
                    feature_id = x.feature_id, 
                    pc = range(len(svd[1]))
                )
            )
            svd = xa.merge([svd, x[['mean', 'scale']]])
            svd = svd.compute()
            return svd

        def gmm(x1, k):
            x2 = GaussianMixture(k, verbose=2).fit(x1.data)
            x3 = xa.Dataset()
            x3['means'] = xa.DataArray(x2.means_, [('clust', range(x2.n_components)), x1.pc])
            x3['covs'] = xa.DataArray(x2.covariances_, [x3.clust, x3.pc, x3.pc.rename(pc='pc1')])
            x3['pred'] = xa.DataArray(x2.predict_proba(x1.data), [x1.cell_id, x3.clust])
            return x3

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
                x1.means.data = sparse.as_coo(x1.means.data)
                x1 = enrichment(x1.means, x1.set)
                return x1

        class _gmm(_enrichment):
            k = 10

            @property
            def svd_k(self):
                return 5*self.k

            @compose(property, lazy, XArrayCache())
            def svd(self):
                print('svd '+str(self.storage))
                svd1 = svd(
                    self.data, 
                    {'cell_id': 1000, 'feature_id': 1000}, 
                    self.svd_k
                )
                return svd1

            @compose(property, lazy, XArrayCache())
            def gmm(self):
                print('gmm '+str(self.storage))
                svd1 = self.svd
                gmm1 = gmm(svd1.u * svd1.s, self.k)
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

            @compose(property, lazy)
            def split(self):
                split = self._split()
                split.prev = self
                return split

        class _item1:
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

        class _split1(_enrichment):
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
            def feature_entrez(self):
                return self.prev.feature_entrez

            class _item(_item1, _gmm):
                pass
            
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

        _gmm._split = _split1

        class _gmm1(_gmm):
            storage = self.storage/'clust1'
            prev = self

            @compose(property, lazy)
            def data(self):
                return self.prev.data.log1p_rpk.todense()

            @property
            def feature_entrez(self):
                return self.prev.feature_entrez

        return _gmm1().split

analysis = _analysis()

self = analysis

# %%
x = xa.merge([analysis.data, analysis.clust1.prev.clust], join='inner')

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
print(
    plot_table(x1)
)


# %%
x = xa.merge([
    analysis.data, 
    analysis.clust1.clust,
    analysis.clust1.means.rename('clust_means'),
    analysis.clust1.enrich.rename({k: 'sig_'+k for k in analysis.clust1.enrich.keys()})
], join='inner')

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

x1 = x1.sort_values('resid', ascending=False).\
    drop_duplicates('pred').\
    query('resid>5').\
    sort_values('table')
    
x1[x1['cell_integrated_snn_res.0.3']==4]
    

# %%
x1 = x.sel(feature_id='CD274').drop_dims(['clust', 'sig'])
x1 = x1.todense()
x1 = xa.merge([x1.drop_dims('umap_dim'), x1.umap.to_dataset('umap_dim')])
x1 = x1.to_dataframe()
x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype('category')
x1['q3'] = quantile(x1.log1p_rpk, q=3)

x2 = x1[['pred', 'q3']]
x2 = sm.stats.Table.from_data(x2)
x2 = pd.concat([
    v.stack().rename(k)
    for k, v in [
        ('table', x2.table_orig), 
        ('resid', x2.resid_pearson), 
        ('fit', x2.fittedvalues)
    ]
], axis=1).reset_index()

x3 = x2.sort_values('resid', ascending=False).\
    query('resid>5')
x3 = x3[x3.q3==pd.Interval(2.245, 4.185)]

x4 = x1[x1.pred.isin([97])]
print(
    ggplot(x4)+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(x4[x4.log1p_rpk==0], color='gray', alpha=0.1)+
        geom_point(x4[x4.log1p_rpk>0], aes(color='q3'), alpha=0.1)+
        theme(legend_position='none')
)

# %%
x1 = x[['sig_coef', 'sig_p']].sel(clust=0).to_dataframe()
x1 = x1.sort_values('sig_p')
x1 = x1[x1.sig_p<1e-4]

# %%
import scipy.cluster.hierarchy as hier

x1 = x.sig_se.transpose('sig', 'clust')
x1 = x1.fillna(0)
#x1 = x1.sel(sig=x1.sig_prefix.isin(['HALLMARK', 'KEGG1']))
x2 = hier.linkage(
    x1.data, method='average', metric='euclidean'
)
x3 = hier.dendrogram(x2, 10, truncate_mode='level')
plt.show()

x1['sig_clust'] = 'sig', hier.fcluster(x2, 100, criterion='maxclust')

x4 = x1['sig_clust'].to_series().reset_index()

x4[x4.sig.str.contains('INTERFER')]
x4 = x4[x4.sig_clust==34]
# %%
