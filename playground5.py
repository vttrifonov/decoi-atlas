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
from ._helpers import config, plot_table


# %%
class _analysis:
    storage = config.cache/'playground5'

    @compose(property, lazy)
    def data(self):
        x = data.c2_neutrophils_integrated.copy()
        x1 = x.counts.copy()
        x1.data = x1.data.asformat('coo')
        x['total'] = x1.sum(dim='feature_id').todense()
        x1 = 1e4*x1/x.total
        x['log1p_rpk'] = np.log1p(x1)
        x['mean'] = x.log1p_rpk.mean(dim='cell_id').todense()
        x['std'] = (x.log1p_rpk**2).mean(dim='cell_id').todense()
        x['std'] = (x['std']-x['mean']**2)**0.5
        x = x.sel(feature_id=x['mean']>0)
        return x

    @compose(property, lazy)
    def clust1(self):
        import dask.array as da
        from sklearn.mixture import GaussianMixture

        def svd(x, chunks, k):
            x = x.chunk(chunks)
            x = (x-x.mean(dim='cell_id'))/x.std(dim='cell_id')
            x = x/np.sqrt(x.sizes['cell_id'])
            x = x.transpose('cell_id', 'feature_id')
            x1 = np.isnan(x).sum(axis=0).compute()
            x = x.sel(feature_id=x1==0)
            svd = da.linalg.svd_compressed(x.data, k=k, n_power_iter=2)
            rand_svd = da.linalg.svd_compressed(
                da.apply_along_axis(
                    np.random.permutation, 0, x.data, 
                    dtype=x.dtype, shape=(svd[0].shape[0],)
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
            svd = svd.compute()
            return svd

        def gmm(x1, k):
            x2 = GaussianMixture(k, verbose=2).fit(x1.data)
            x3 = xa.Dataset()
            x3['means'] = xa.DataArray(x2.means_, [('clust', range(x2.n_components)), x1.pc])
            x3['covs'] = xa.DataArray(x2.covariances_, [x3.clust, x3.pc, x3.pc.rename(pc='pc1')])
            x3['pred'] = xa.DataArray(x2.predict_proba(x1.data), [x1.cell_id, x3.clust])
            return x3

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

        class _split1:
            @property
            def storage(self):
                return self.prev.storage/'split'

            @property
            def k(self):
                return 10*self.prev.k

            @property
            def data(self):
                return self.prev.data

            class _item(_item1, _gmm):
                pass

            def item(self, c):
                item = self._item()
                item.prev = self
                item.c = c
                return item
            
            @property
            def clust(self):
                c = self.prev.clust.to_series().drop_duplicates()
                c = [self.item(c).clust for c in c]
                c1 = np.cumsum([c.max().data+1 for c in c])
                c1 = [0]+list(c1[:-1])
                c = [c+c1 for c, c1 in zip(c, c1)]
                c = xa.concat(c, dim='cell_id')
                return c 

        _gmm._split = _split1            

        self = analysis
        class _gmm1(_gmm):
            storage = self.storage/'clust1'
            prev = self

            @compose(property, lazy)
            def data(self):
                return self.prev.data.log1p_rpk.todense()
        self = _gmm1()
        x, chunks, k = (
            self.data, 
            {'cell_id': 1000, 'feature_id': 1000}, 
            self.svd_k
        )

        return _gmm1().split

analysis = _analysis()

self = analysis

# %%
x = xa.merge([analysis.data, analysis.gmm1.clust], join='inner')

# %%
x1 = x[['cell_integrated_snn_res.0.3', 'pred']].to_dataframe()
for c in x1:
    x1[c] = x1[c].astype('category')
x1 = sm.stats.Table.from_data(x1)
print(
    plot_table(x1, show_obs=False, show_exp=False)
)

x3 = 3
x2 = x1.resid_pearson.loc[x3,:]
x2 = list(x2[x2>5].index.astype(int))
x2 = x1.resid_pearson.loc[:,x2]
x2 = x2[x2.index!=x3]
x2 = x2.max(axis=0)
x2 = list(x2[x2<5].index.astype(int))
len(x2)

x1 = x[['cell_integrated_snn_res.0.3', 'pred']].to_dataframe()
x1['cell_integrated_snn_res.0.3']=x1['cell_integrated_snn_res.0.3']==x3
x1['pred'] = x1['pred'].isin(x2)
x1 = sm.stats.Table.from_data(x1)
print(
    plot_table(x1)
)

# %%
x1 = xa.merge([x[['pred', 'cell_integrated_snn_res.0.3']], x.umap.to_dataset(dim='umap_dim')]).to_dataframe()
x1['pred'] = x1.pred.astype('category')
print(
    ggplot(x1[x1['pred'].isin(x2)])+
        aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='pred'), alpha=0.1)+
        theme(legend_position='none')
)

# %%
x2 = x1.pred.value_counts()
# %%
