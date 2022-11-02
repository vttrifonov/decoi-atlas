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

        x2 = (x.total_cell/x.total_cell.sum())*x.total_feature
        #x2 = x2.persist()
        x['resid'] = (x1-x2)/np.sqrt(x2)
        x = x.compute()
        x = x.sel(feature_id=x['total_feature']>0)

        return x

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

        class _gmm1(_gmm):
            storage = self.storage/'clust1'
            prev = self

            @compose(property, lazy)
            def data(self):
                return self.prev.data.log1p_rpk.todense()

        return _gmm1().split

analysis = _analysis()

self = analysis

# %%
def moments():
    from scipy.stats import kurtosis, skew
    x1 = np.random.randn(1000)
    x2 = [(x1**k).mean() for k in range(5)]
    x3 = np.zeros((len(x2),))
    x3[0] = 1
    for n in range(2, len(x3)):
        c = 1
        for j in range(n+1):
            x3[n] += c*x2[n-j]
            c = -(c*x2[1]*(n-j))/(j+1)

    x4 = x3[3:]/[x3[2]**(k/2) for k in range(3, len(x3))]
    x4 - [skew(x1), kurtosis(x1, fisher=False)]

# %%
x = xa.merge([analysis.data, analysis.clust1.prev.clust], join='inner')

# %%
x1 = x.total_cell.to_dataframe()
print(
    ggplot(x1)+aes('total_cell')+geom_freqpoly(bins=100)
)

# %%
x1 = x[['mean', 'std']].to_dataframe()
print(
    ggplot(x1)+aes('std')+geom_freqpoly(bins=100)
)

print(
    ggplot(x1)+aes('mean', 'std')+geom_point()
)

# %%
x1 = x.sel(feature_id=['IFITM2', 'C5AR1'])
x1 = x1.todense()
x1 = xa.merge([
    x1.drop_dims(['umap_dim', 'feature_id']),
    x1.umap.to_dataset('umap_dim'),
    x1.resid.to_dataset('feature_id'),
])
x1 = x1.to_dataframe()
for x2 in ['IFITM2', 'C5AR1']:
    x1[x2+'_q3'] = quantile(x1[x2], q=3)

print(
    ggplot(x1)+
        aes('IFITM2', 'C5AR1')+
        geom_point()+geom_density_2d(color='red')+
        facet_grid('~cell_diagnosis')+
        theme(figure_size=(7, 2))
)

x2 = x1[x1['cell_integrated_snn_res.0.3']==2]
x2 = x2[['IFITM2_q3', 'C5AR1_q3']]
x2 = sm.stats.Table.from_data(x2)
print(
    plot_table(x2)
)

# %%
x1 = x.sel(feature_id='C5AR1')
x1 = x1.todense()
x1 = xa.merge([x1.drop_dims('umap_dim'), x1.umap.to_dataset('umap_dim')])
x1 = x1.to_dataframe()
x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype('category')
x1['q3'] = pd.qcut(x1.resid, q=3)

print(
    ggplot(x1)+aes('UMAP_1', 'UMAP_2')+
        geom_point(aes(color='q3'), alpha=0.1)
)

x2 = x1[['cell_integrated_snn_res.0.3', 'q3']]
x2 = sm.stats.Table.from_data(x2)
print(
    plot_table(x2)
)

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
