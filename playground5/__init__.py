# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground5'

# %%
import numpy as np
import pandas as pd
import xarray as xa
import ipywidgets as widgets
from IPython.display import display
import statsmodels.api as sm
from plotnine import *
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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
@property
def _analysis_cell_type_clusters(self):
    x1 = xa.merge([
        self.data[['cell_integrated_snn_res.0.3']],
        self.data.umap.to_dataset('umap_dim')
    ], join='inner')
    x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype(str)
    x1 = x1.to_dataframe().reset_index()
    return x1
_analysis.cell_type_clusters = _analysis_cell_type_clusters

#%%
@property
def _analysis_clust3_cell_type_clusters(self):
    x1 = xa.merge([
        self.prev.data.umap.to_dataset('umap_dim'),
        self.clust.rename('clust')
    ], join='inner')
    x1['clust'] = x1['clust'].astype(str)
    x1 = x1.to_dataframe().reset_index()
    return x1
_analysis._clust3.cell_type_clusters = _analysis_clust3_cell_type_clusters

#%%
@property
def _analysis_cluster_to_cell_overlap(self):
    def cluster_to_cell_overlap(
        num_clusters=20
    ):
        x1 = xa.merge([
            self.data[['cell_integrated_snn_res.0.3']],
            self.clust3(num_clusters).clust.rename('clust')
        ], join='inner')
        x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype(str)
        x1['clust'] = x1['clust'].astype(str)
        display(
            plot_table1(
                x1[['cell_integrated_snn_res.0.3', 'clust']].to_dataframe()
            )
        )
    cluster_to_cell_overlap = widgets.interact(
        cluster_to_cell_overlap,
        num_clusters=[10, 20, 30, 40]    
    )
    return cluster_to_cell_overlap
_analysis.cluster_to_cell_overlap = _analysis_cluster_to_cell_overlap

# %%
@property
def _analysis_clust3_enrichment_clust1_cell_cluster_sigs(self):
    x1 = xa.merge([
        self.prev.sigs.set.sum(dim='gene').rename('sig_size').todense(),
        self.data.rename(clust1='cell_clust')
    ])
    x1 = x1.to_dataframe().reset_index()
    x1['sig'] = x1.sig.str.replace('^[^_]*_', '', regex=True)

    @widgets.interact(
        sig_prefix=[''] + list(x1.sig_prefix.drop_duplicates()), 
        cell_clust=[''] + list(np.sort(x1.cell_clust.drop_duplicates()).astype(str)),
        sig='',
        sig_size = widgets.IntRangeSlider(value=[10, 500], min=1, max=x1.sig_size.max()),
        coef = ['ascending', 'descending'],
        rows=(0, 100, 20)
    )
    def cell_cluster_sigs(
        sig_prefix='', 
        cell_clust='', 
        sig='', 
        sig_size=[10,500],
        coef='descending', 
        rows=20
    ):    
        x5 = x1
        x5 = x5[x5.sig_size>=sig_size[0]]
        x5 = x5[x5.sig_size<=sig_size[1]]
        if sig_prefix!='':
            x5 = x5[x5.sig_prefix==sig_prefix]
        if cell_clust!='':
            x5 = x5[x5.cell_clust==int(cell_clust)]
        if sig!='':
            x5 = x5[x5.sig.str.contains(sig, regex=True)]

        pd.set_option('display.max_rows', rows)
        return x5.sort_values('coef', ascending=(coef=='ascending')).head(rows)

    return cell_cluster_sigs
_analysis._clust3._enrichment._clust1.cell_cluster_sigs = _analysis_clust3_enrichment_clust1_cell_cluster_sigs

# %%
@property
def _analysis_clust3_enrichment_clust1_pca1(self):
    x1 = self
    x3 = x1.svd
    x3 = x3.u * x3.s
    x3 = x3.sel(pc=x3.pc<2)
    x3['pc'] = xa.DataArray(['x', 'y'], [('pc', [0,1])])[x3.pc]
    x3 = x3.to_dataset('pc')
    x3 = xa.merge([x3, x1.clust.rename('clust')])
    x3 = x3.to_dataframe()
    x3['clust'] = x3.clust.astype(str)
    return x3
_analysis._clust3._enrichment._clust1.pca1 = _analysis_clust3_enrichment_clust1_pca1

# %%
@property
def _analysis_clust3_enrichment_clust1_pca2(self):
    x4 = self.svd.rename(clust1='cell_clust')
    x4['cell_clust'] = x4.cell_clust.astype('str')[x4.cell_clust]

    _, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3))

    for a in [a2, a3]:
        a.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )

    x4['s'].query(pc='pc<19').to_series().\
        pipe(np.log2).\
        plot(style='.', ax=a1)

    x4['rand_s'].query(pc='pc<19').to_series().\
        pipe(np.log2).\
        plot(style='.', ax=a1)
        

    x4['v'].sel(pc=0).to_series().sort_values().plot(ax=a2)

    x4['u'].sel(pc=0).to_series().sort_values().plot(ax=a3)

_analysis._clust3._enrichment._clust1.pca2 = _analysis_clust3_enrichment_clust1_pca2

# %%
@property
def _analysis_clust3_enrichment_clust1_clust_sizes(self):
    x1 = self.clust.rename('sig_clust').to_dataframe().\
        groupby('sig_clust').size().sort_values().rename('num_sigs').reset_index()
    x1['sig_clust'] = x1.sig_clust.astype(pd.CategoricalDtype(x1.sig_clust))
    return x1
_analysis._clust3._enrichment._clust1.clust_sizes = _analysis_clust3_enrichment_clust1_clust_sizes


# %%
@property
def _analysis_clust3_enrichment_clust1_pca3(self):
    x1 = self
    x3 = x1.svd
    x3 = x3.u * x3.s
    x3 = x3.sel(pc=x3.pc<2)
    x3['pc'] = xa.DataArray(['x', 'y'], [('pc', [0,1])])[x3.pc]
    x3 = x3.to_dataset('pc')
    x3 = xa.merge([x3, x1.clust.rename('clust')])
    x3 = x3.to_dataframe().reset_index()
    x3['clust'] = x3.clust.astype(str)
    return x3
_analysis._clust3._enrichment._clust1.pca3 = _analysis_clust3_enrichment_clust1_pca3

# %%
@property
def _analysis_clust3_enrichment_clust1_clust_covs(self):
    x1 = self.prev.prev
    x2 = self
    x5 = x2.gmm
    x5['clust'] = x5.clust.astype('str')[x5.clust]
    x6 = np.linalg.svd(x5.covs.data, compute_uv=False)
    x6 = xa.DataArray(x6, [x5.clust, ('pc', range(x6.shape[1]))], name='s')
    x6 = xa.merge([
        x6,
        (x5.means**2).sum(dim='pc'),
        x2.data.shape[0]*x2.gmm.weights
    ])
    x6 = (x6.s*x6.weights).rename('s').to_dataframe().reset_index()
    return x6
_analysis._clust3._enrichment._clust1.clust_covs = _analysis_clust3_enrichment_clust1_clust_covs

# %%
@property
def _analysis_clust3_enrichment_clust1_clust_means(self):
    x1 = self.prev.prev
    x2 = self
    x5 = x2.gmm.rename(clust='sig_clust')
    x4 = x2.svd.rename(clust1='cell_clust')
    x6 = (x5.means@x4.v)*x4.scale + x4['mean']
    x6 = x6.rename('means').to_dataframe().reset_index()

    x7 = xa.merge([
        x1.prev.data[['cell_integrated_snn_res.0.3']],
        x1.clust.rename('cell_clust')
    ], join='inner')
    x7 = sm.stats.Table.from_data(x7.to_dataframe())
    x7 = x7.resid_pearson.melt(ignore_index=False).reset_index()
    x7 = x7.sort_values('value', ascending=False).\
        drop_duplicates(x7.columns[1]).\
        sort_values([x7.columns[0], 'value']).cell_clust

    x8 = x6.pivot_table(index='sig_clust', columns='cell_clust', values='means')
    x8 = linkage(x8, method='average')
    x8 = dendrogram(x8, no_plot=True)['leaves']

    x6['cell_clust'] = x6.cell_clust.astype(pd.CategoricalDtype(x7))
    x6['sig_clust'] = x6.sig_clust.astype(pd.CategoricalDtype(x8))
    return x6
_analysis._clust3._enrichment._clust1.clust_means = _analysis_clust3_enrichment_clust1_clust_means

# %%
@compose(property, lazy)
def _analysis_clust3_enrichment_clust1_data1(self):
    x1 = self.prev.prev
    x2 = self

    x3 = xa.merge([
        x1.prev.data.drop_dims(['feature_id', 'umap_dim']),
        x1.prev.data.umap.to_dataset(dim='umap_dim'),
        x1.enrich.sigs.drop('sig_prefix').rename(means='clust_means', clust='cell_clust_id'),
        x1.clust.rename('cell_clust'),
        x1.gmm.pred.rename('cell_proba').rename(clust='cell_clust_id'),
        x2.data.sig_prefix.reset_coords(),
        x2.clust.rename('sig_clust'),
        x2.gmm.pred.rename('sig_proba').rename(clust='sig_clust_id'),
        x2.means.rename(clust='sig_clust_id', clust1='cell_clust_id').rename('sig_clust_means')
    ], join='inner')

    x3 = x3.sel(feature_id=x3.feature_entrez.sum(dim='gene').todense()>0)
    x3['set'] = xa.apply_ufunc(
        np.matmul, x3.feature_entrez, x3.set,
        input_core_dims=[['feature_id', 'gene'], ['gene', 'sig']],
        output_core_dims=[['feature_id', 'sig']]
    )>0
    x3['clust_means'] = xa.apply_ufunc(
        np.matmul, x3.feature_entrez, x3.clust_means,
        input_core_dims=[['feature_id', 'gene'], ['gene', 'cell_clust_id']],
        output_core_dims=[['feature_id', 'cell_clust_id']]
    )
    x3['clust_means'] = x3.clust_means/x3.feature_entrez.sum(dim='gene').todense()

    x3 = x3.drop_dims('gene')
    return x3
_analysis._clust3._enrichment._clust1.data1 = _analysis_clust3_enrichment_clust1_data1

# %%
@property
def _analysis_clust3_enrichment_clust1_sigs_for_clust(self):
    x3 = self.data1
    x4 = xa.merge([
        x3.sig_prefix,
        x3.sig_proba.rename(sig_clust_id='sig_clust'),
        x3.set.sum(dim='feature_id').todense().rename('sig_size')
    ])
    x4 = x4.to_dataframe().reset_index()
    x4['sig'] = x4.sig.str.replace('^[^_]*_', '', regex=True)

    @widgets.interact(
        sig_prefix=[''] + list(x4.sig_prefix.drop_duplicates()),
        sig_clust=[''] + list(x4.sig_clust.drop_duplicates().astype(str)),    
        sig='',
        sig_size = widgets.IntRangeSlider(value=[10, 500], min=1, max=x4.sig_size.max()),
        sig_proba = (0, 1, 0.1),
        rows=(0, 100, 20)
    )
    def sigs_for_clust(
        sig_prefix='', 
        sig_clust='', 
        sig='',
        sig_size=[10, 500],
        sig_proba=0.9,
        rows=20
    ):
        x5 = x4
        x5 = x5[x5.sig_size>=sig_size[0]]
        x5 = x5[x5.sig_size<=sig_size[1]]
        x5 = x5[x5.sig_proba>=sig_proba]
        if sig_prefix!='':
            x5 = x5[x5.sig_prefix==sig_prefix]
        if sig_clust!='':
            x5 = x5[x5.sig_clust==int(sig_clust)]
        if sig!='':
            x5 = x5[x5.sig.str.contains(sig, regex=True)]
        print(f'{x5.shape[0]} gene sets.')
        pd.set_option('display.max_rows', rows)
        return x5.sort_values('sig_proba', ascending=False).head(rows)

    return sigs_for_clust
_analysis._clust3._enrichment._clust1.sigs_for_clust = _analysis_clust3_enrichment_clust1_sigs_for_clust

# %%
@property
def _analysis_clust3_enrichment_clust1_expr_for_clust(self):
    x1 = self.prev.prev
    x3 = self.data1

    x7 = xa.merge([
        x1.prev.data[['cell_integrated_snn_res.0.3']],
        x1.clust.rename('cell_clust')
    ], join='inner')
    x7 = sm.stats.Table.from_data(x7.to_dataframe())
    x7 = x7.resid_pearson.melt(ignore_index=False).reset_index()
    x7 = x7.sort_values('value', ascending=False).\
        drop_duplicates(x7.columns[1]).\
        sort_values([x7.columns[0], 'value']).cell_clust

    sig_prefix = widgets.Dropdown(description='sig_prefix', options=['']+list(np.unique(x3.sig_prefix.data)))
    sig_regex = widgets.Text(description='sig_regex')
    sig = widgets.Dropdown(description='sig', options=['']+list(np.unique(x3.sig.data)))
    gene_regex = widgets.Text(description='gene_regex')
    genes = widgets.SelectMultiple(
        description='genes',
        options=np.sort(x3.feature_id.data)
    )
    sig_button = widgets.Button(description='update sigs')
    genes_button = widgets.Button(description='update genes')
    plot_button = widgets.Button(description='update plot')
    out = widgets.Output()

    def sig_update(sig_prefix, sig_regex, genes):    
        x4 = x3[['sig_prefix', 'sig']].to_dataframe().reset_index()

        if sig_prefix!='':
            x4 = x4[x4.sig_prefix==sig_prefix]

        if sig!='':
            x4 = x4[x4.sig.str.contains(sig_regex, regex=True)]

        x4 = x4.sig.to_list()

        if genes != []:        
            x5 = x3.set.sel(sig=x4)
            x5 = x5.sel(feature_id=genes)
            x5 = x5.sum(dim='feature_id')==x5.sizes['feature_id']
            x4 = list(x5.sel(sig=x5.todense()).sig.data)

        x4 = [''] + x4

        sig.options = x4
        sig.value = ''

    def genes_update(sig, gene_regex):
        if sig=='':
            x4 = x3.feature_id.data
        else:
            x4 = x3.set.sel(sig=sig).todense()
            x4 = x4.sel(feature_id=x4).feature_id.data

        if gene_regex != '':
            x4 = pd.Series(x4)
            x4 = x4[x4.str.contains(gene_regex, regex=True)].to_list()
            
        x4 = np.sort(x4)

        genes.options = x4
        genes.value = genes.options

    genes_button.on_click(lambda b: genes_update(sig.value, gene_regex.value))
    sig_button.on_click(lambda b: sig_update(sig_prefix.value, sig_regex.value, list(genes.value)))

    def plot(sig, genes):
        x4 = x3[['set', 'clust_means']]
        x4 = x4.rename(cell_clust_id='cell_clust')

        if sig!='':
            x4 = x4.sel(sig=sig)
            x4['set'] = x4.set.todense()
            x4 = x4.to_dataframe().reset_index()    
            x4['cell_clust'] = x4.cell_clust.astype(pd.CategoricalDtype(x7))

            p =  (
                ggplot(x4)+aes('cell_clust', 'clust_means', color='set')+
                    geom_boxplot()+
                    labs(title=sig)
            )
            if len(genes)>0:
                p = p + geom_point(data=x4[x4.feature_id.isin(genes)], color='black')
            display(p)

            x9 = x4[x4.set==True].copy()
            x9 = x9[x9.feature_id.isin(genes)]
        else:
            if len(genes)==0:
                return
            x9 = x4.clust_means.sel(feature_id=genes)
            x9 = x9.to_dataframe().reset_index()    
            x9['cell_clust'] = x9.cell_clust.astype(pd.CategoricalDtype(x7))

        x5 = x9.feature_id.drop_duplicates().shape[0]
        if x5>1:
            x10 = x9.pivot_table(index='feature_id', columns='cell_clust', values='clust_means')
            x8 = linkage(x10, method='average')
            x8 = dendrogram(x8, no_plot=True)['leaves']
            x8 = x10.index.to_numpy()[x8]    
            x9['feature_id'] = x9.feature_id.astype(pd.CategoricalDtype(x8))

        display(widgets.Label(f'{x5} selected gene.'))
        if x5>0 and x5<500:
            display(
                ggplot(x9)+aes('cell_clust', 'feature_id', fill='clust_means')+
                    geom_tile()+
                    scale_fill_gradient2(
                        low='blue', mid='white', high='red',
                        midpoint=0
                    )+
                    theme(
                        figure_size=(5, 0.2*x5),
                    )+
                    labs(y='')
            )

    plot_button.on_click(out.capture(clear_output=True)(lambda b: plot(sig.value, list(genes.value))))

    return widgets.VBox([
        widgets.VBox([
            sig_prefix,
            widgets.HBox([sig_regex, sig, sig_button]),
            widgets.HBox([gene_regex, genes, genes_button]),
            plot_button
        ]),  
        out
    ])
_analysis._clust3._enrichment._clust1.expr_for_clust = _analysis_clust3_enrichment_clust1_expr_for_clust

# %%
def plot_table1(x1):
    cell, clust = x1.columns[0], x1.columns[1]
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
        drop_duplicates(clust).\
        sort_values([cell, 'resid'])

    x4 = list(x3[cell].drop_duplicates())
    x4 = x4+list(set(x1[cell])-set(x4))
    x1[cell] = x1[cell].astype(pd.CategoricalDtype(x4, ordered=True))
    x1[clust] = x1[clust].astype(pd.CategoricalDtype(x3[clust].drop_duplicates(), ordered=True))
    x1['label'] = x1['table'].astype(str) + '\n' + x1['fit'].astype(int).astype(str)
    return (
        ggplot(x1)+
            aes(cell, clust)+
            geom_tile(aes(fill='resid'))+
            geom_text(aes(label='label'), size=7)+
            scale_fill_gradient2(
                low='blue', mid='white', high='red',
                midpoint=0
            )+
            theme(figure_size=(5, 6))
    )