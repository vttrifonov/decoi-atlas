# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

# %%
from re import X
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *

from .common import helpers
from .common.caching import compose, lazy, XArrayCache, CSVCache
from ._data import data
from ._helpers import config

# %%


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

    @compose(property, lazy, XArrayCache())
    def svd(self):
        import dask.array as da

        x = self.data[['log1p_rpk', 'mean', 'std']].todense()
        x = x.chunk({'cell_id': 1000, 'feature_id': 1000})
        x = (x.log1p_rpk-x['mean'])/x['std']
        x = x/np.sqrt(x.sizes['cell_id'])
        x = x.transpose('cell_id', 'feature_id')
        svd = da.linalg.svd_compressed(x.data, k=2000, n_power_iter=2)
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

    @compose(property, lazy, XArrayCache())
    def gmm(self):
        from sklearn.mixture import GaussianMixture

        x = self.svd
        x = x.query(pc='pc<500')
        x = x.u * x.s
        x1 = GaussianMixture(100, verbose=2).fit(x.data)
        x2 = xa.Dataset()
        x2['means'] = xa.DataArray(x1.means_, [('clust', range(x1.n_components)), x.feature_id])
        x2['covs'] = xa.DataArray(x1.covariances_, [x2.clust, x.feature_id, x.feature_id.rename(feature_id='feature_id1')])
        x2['prec'] = xa.DataArray(x1.precisions_, [x2.clust, x.feature_id, x.feature_id.rename(feature_id='feature_id1')])
        x2['prec_chol'] = xa.DataArray(x1.precisions_cholesky_, [x2.clust, x.feature_id, x.feature_id.rename(feature_id='feature_id1')])
        x2['pred'] = xa.DataArray(x1.predict_proba(x.data), [x.cell_id, x2.clust])
        return x2

    @compose(property, lazy)
    def svd1_data(self):
        x = self.data
        x = x[['cell_donor', 'log1p_rpk', 'mean', 'std']].todense()
        x1 = (x.log1p_rpk-x['mean'])/x['std']
        x1 = x1/np.sqrt(x1.sizes['cell_id'])
        x1 = x1.transpose('cell_id', 'feature_id')
        x['log1p_rpk'] = x1
        return x

    def svd1(self):
        storage = self.storage/'svd1'
        import dask.array as da
        def svd1(x1):
            x1 = x1.chunk({'cell_id': 1000, 'feature_id': 1000})
            svd = da.linalg.svd_compressed(
                x1.data, 
                k = min(x1.shape), n_power_iter=2
            )
            rand_svd = da.linalg.svd_compressed(
                da.apply_along_axis(
                    np.random.permutation, 0, x1.data, 
                    dtype=x1.dtype, shape=(svd[0].shape[0],)
                ),
                k = min(x1.shape), n_power_iter=2
            )
            svd = da.compute(svd[1], rand_svd[1])
            k = (svd[0]>svd[1]).sum()
            svd = da.linalg.svd_compressed(
                x1.data, 
                k = k, n_power_iter=2
            )
            rand_svd = da.linalg.svd_compressed(
                da.apply_along_axis(
                    np.random.permutation, 0, x1.data, 
                    dtype=x1.dtype, shape=(svd[0].shape[0],)
                ),
                k = k, n_power_iter=2
            )
            svd = xa.Dataset(
                dict(
                    u=(('cell_id', 'pc'), svd[0]),
                    s=(('pc',), svd[1]),
                    v=(('feature_id', 'pc'), svd[2].T),
                    rand_s=(('pc',), rand_svd[1])
                ),
                dict(
                    cell_id = x1.cell_id, 
                    feature_id = x1.feature_id, 
                    pc = range(len(svd[1]))
                )
            )
            svd = svd.compute()
            return svd

        x = self.data
        x = x[['cell_donor', 'log1p_rpk', 'mean', 'std']].todense()
        x1 = (x.log1p_rpk-x['mean'])/x['std']
        x1 = x1/np.sqrt(x1.sizes['cell_id'])
        x1 = x1.transpose('cell_id', 'feature_id')
        x['log1p_rpk'] = x1

analysis = _analysis()

self = analysis

# %%
svd = self.svd
x1 = svd[['s', 'rand_s']].to_dataframe().reset_index()
print((
    (x1.s**2).sum()/svd.sizes['feature_id'],
    (x1.rand_s**2).sum()/svd.sizes['feature_id']
))
print(
    ggplot(x1.iloc[:500,:])+aes('s', 'rand_s')+
        geom_point()+
        geom_abline(slope=1, intercept=0)
)
print(
    ggplot(x1)+aes('np.log10(s)', 'np.log10(s/rand_s)')+
        geom_point()
)


# %%
