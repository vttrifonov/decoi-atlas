# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground5'

# %%
import numpy as np
import xarray as xa

from ..common import helpers
from ..common.caching import compose, lazy, XArrayCache
from .._data import data
from .._helpers import config

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
    x3['weights'] = xa.DataArray(x2.weights_, [x3.clust])
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
    @compose(property, lazy)
    def sigs(self):
        from ..sigs._sigs import sigs 
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
        return x1

    @compose(property, lazy, XArrayCache())
    def data(self):
        from ..sigs.fit import enrichment
        import sparse
        
        x1 = self.sigs
        x1.means.data = sparse.as_coo(x1.means.data)
        x1 = enrichment(x1.means, x1.set)
        return x1

class _gmm:
    k = 10
    dims = ['cell_id', 'feature_id']

    @property
    def svd_k(self):
        return 5*self.k

    @compose(property, lazy, XArrayCache())
    def svd(self):
        print('svd '+str(self.storage))
        svd1 = svd(
            self.data, 
            k = self.svd_k,
            chunks = {self.dims[0]: 1000, self.dims[1]: 1000},
            dims = self.dims
        )
        return svd1

    @compose(property, lazy, XArrayCache())
    def gmm(self):
        print('gmm '+str(self.storage))
        svd1 = self.svd
        gmm1 = gmm(svd1.u * svd1.s, self.k, dims=[self.dims[0], 'pc'])
        return gmm1

    @compose(property, lazy)
    def kl(self):
        return gmm_kl(self.gmm)

    @compose(property, lazy, XArrayCache())
    def clust(self):
        if self.k==1:
            return xa.DataArray(
                [0]*self.data.sizes[self.dims[0]],
                [self.data[self.dims[0]]]
            )                    
        return self.gmm.pred.argmax(dim='clust')

    @compose(property, lazy, XArrayCache())
    def means(self):
        if self.k==1:
            x = self.data.mean(dim=self.dims[0])
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
    def gmm_svd(self):
        x4 = np.linalg.svd(self.gmm.covs)
        return xa.Dataset(dict(
            u = xa.DataArray(x4[0], [self.gmm.clust, self.gmm.pc, self.gmm.pc_1]),
            s = xa.DataArray(x4[1], [self.gmm.clust, self.gmm.pc_1])
        ))

    def proba(self, X):    
        x4 = self.gmm_svd
        x4 = xa.merge([
            (x4.u * 1/np.sqrt(x4.s)).rename('mat'),
            -0.5*np.log(x4.s).sum(dim='pc_1').rename('log_det')
        ])
        sigma = x4
        
        x5 = xa.dot(X - self.gmm.means, sigma.mat, dims=['pc'])
        x5 = (x5**2).sum(dim='pc_1')
        x5 = -0.5*(np.log(2*np.pi)*X.sizes['pc']+x5)+sigma.log_det
        x5 = x5 + np.log(self.gmm.weights)
        x5 = x5 - x5.max(dim='clust')
        x5 = np.exp(x5)
        x5 = x5/x5.sum(dim='clust')
        return x5

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
        from ..sigs.entrez import symbol_entrez

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
        class _gmm1(_gmm):
            storage = self.storage/'clust2'
            prev = self
            k = 100

            @compose(property, lazy)
            def data(self):
                return self.prev.data.log1p_rpk.todense()

            @property
            def feature_entrez(self):
                return self.prev.feature_entrez

            class _enrichment(_enrichment):
                @property
                def storage(self):
                    return self.prev.storage/'enrich'

                class _clust1(_gmm):
                    @property
                    def storage(self):
                        return self.prev.storage/'clust1'/str(self.k)

                    @property
                    def data(self):
                        return self.prev.data.coef.fillna(0).\
                            transpose('sig', 'clust').rename(clust='clust1')

                    dims = ['sig', 'clust1']
                    @property
                    def svd_k(self):
                        return min(self.data.sizes.values())

                def clust1(self, k):
                    clust = self._clust1()
                    clust.k = k
                    clust.prev = self              
                    return clust

            @compose(property, lazy)
            def enrich(self):
                enrich = self._enrichment()
                enrich.prev = self
                return enrich

        return _gmm1()

    class _clust3(_gmm):
        def __init__(self, prev, t):
            self.k = t
            self.storage = prev.storage/'clust3'/str(t)
            self.prev = prev

        @compose(property, lazy)
        def data(self):
            return self.prev.data.log1p_rpk.todense()

        @property
        def feature_entrez(self):
            return self.prev.feature_entrez

        class _enrichment(_enrichment):
            @property
            def storage(self):
                return self.prev.storage/'enrich'

            @property
            def feature_entrez(self):
                return self.prev.feature_entrez

            @property
            def means(self):
                return self.prev.means

            class _clust1(_gmm):
                @property
                def storage(self):
                    return self.prev.storage/'clust1'/str(self.k)

                @property
                def data(self):
                    return self.prev.data.coef.fillna(0).\
                        transpose('sig', 'clust').rename(clust='clust1')

                dims = ['sig', 'clust1']
                @property
                def svd_k(self):
                    return min(self.data.sizes.values())

            def clust1(self, k):
                clust = self._clust1()
                clust.k = k
                clust.prev = self              
                return clust

        @compose(property, lazy)
        def enrich(self):
            enrich = self._enrichment()
            enrich.prev = self
            return enrich

    def clust3(self, t):
        return self._clust3(self, t)

analysis = _analysis()

self = analysis

# %%
