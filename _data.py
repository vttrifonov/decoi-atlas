#%%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

# %%
import xarray as xa
import numpy as np
from ._helpers import config
from .common.caching import compose, lazy, XArrayCache

# %%
class _data:
    storage = config.cache / 'data'

    @compose(property, lazy, XArrayCache())
    def c1_pbmc(self):
        from rpy2.robjects import r as R
        from rpy2.robjects import pandas2ri, numpy2ri
        import sparse

        R((
            f'source("{config.root}/_data.R");'
            'd <- data$c1_pbmc'
        ))

        x1 = R('d@meta.data')
        x1 = pandas2ri.rpy2py(x1)
        x1.index.names = ['cell_id']
        x1 = x1.where(x1!=R('NA_character_')[0], '')
        x1 = x1.to_xarray()
        x1 = x1.rename({
            k: 'cell_'+k for k in x1.keys()
        })

        x3 = R('d[["RNA"]]@meta.features')
        x3 = pandas2ri.rpy2py(x3)
        x3.index.names = ['feature_id']
        x3 = x3.to_xarray()
        x3 = x3.rename({
            k: 'feature_'+k for k in x3.keys()
        })

        x2 = R((
            'x <- GetAssayData(d[["RNA"]], slot="counts");'
            'list(x@i, x@p, x@x, rownames(x), colnames(x))'
        ))
        x2 = [numpy2ri.rpy2py(x) for _, x in x2.items()]
        x2 = xa.DataArray(
            sparse.GCXS(
                (x2[2], x2[0], x2[1]), 
                shape=(len(x2[3]), len(x2[4])),
                compressed_axes=(1,)
            ),
            [('feature_id', x2[3]), ('cell_id', x2[4])],
            name='counts'
        )

        x5 = R((
            'x<-Embeddings(d[["umap"]]);'
            'list(x, rownames(x), colnames(x))'
        ))
        x5 = [numpy2ri.rpy2py(x) for _, x in x5.items()]
        x5 = xa.DataArray(
            x5[0],
            [('cell_id', x5[1]), ('umap_dim', x5[2])],
            name='umap'
        )

        x4 = xa.merge([x1, x2, x3, x5], join='inner')

        return x4

    @compose(property, lazy, XArrayCache())
    def c2_wb_pbmc(self):
        from rpy2.robjects import r as R
        from rpy2.robjects import pandas2ri, numpy2ri
        import sparse

        R((
            f'source("{config.root}/_data.R");'
            'd <- data$c2_wb_pbmc'
        ))

        x1 = R('d@meta.data')
        x1 = pandas2ri.rpy2py(x1)
        x1.index.names = ['cell_id']
        x1 = x1.where(x1!=R('NA_character_')[0], '')
        x1 = x1.to_xarray()
        x1 = x1.rename({
            k: 'cell_'+k for k in x1.keys()
        })

        x3 = R('d[["RNA"]]@meta.features')
        x3 = pandas2ri.rpy2py(x3)
        x3.index.names = ['feature_id']
        x3 = x3.to_xarray()
        x3 = x3.rename({
            k: 'feature_'+k for k in x3.keys()
        })

        x2 = R((
            'x <- GetAssayData(d[["RNA"]], slot="counts");'
            'list(x@i, x@p, x@x, rownames(x), colnames(x))'
        ))
        x2 = [numpy2ri.rpy2py(x) for _, x in x2.items()]
        x2 = xa.DataArray(
            sparse.GCXS(
                (x2[2], x2[0], x2[1]), 
                shape=(len(x2[3]), len(x2[4])),
                compressed_axes=(1,)
            ),
            [('feature_id', x2[3]), ('cell_id', x2[4])],
            name='counts'
        )

        x5 = R((
            'x<-Embeddings(d[["umap"]]);'
            'list(x, rownames(x), colnames(x))'
        ))
        x5 = [numpy2ri.rpy2py(x) for _, x in x5.items()]
        x5 = xa.DataArray(
            x5[0],
            [('cell_id', x5[1]), ('umap_dim', x5[2])],
            name='umap'
        )

        x4 = xa.merge([x1, x2, x3, x5], join='inner')

        return x4

    @compose(property, lazy, XArrayCache())
    def c2_neutrophils_integrated(self):
        from rpy2.robjects import r as R
        from rpy2.robjects import pandas2ri, numpy2ri
        import sparse

        R((
            f'source("{config.root}/_data.R");'
            'd <- data$c2_neutrophils_integrated'
        ))

        x1 = R('d@meta.data')
        x1 = pandas2ri.rpy2py(x1)
        x1.index.names = ['cell_id']
        x1 = x1.where(x1!=R('NA_character_')[0], '')
        x1 = x1.to_xarray()
        x1 = x1.rename({
            k: 'cell_'+k for k in x1.keys()
        })

        x3 = R('d[["RNA"]]@meta.features')
        x3 = pandas2ri.rpy2py(x3)
        x3.index.names = ['feature_id']
        x3 = x3.to_xarray()
        x3 = x3.rename({
            k: 'feature_'+k for k in x3.keys()
        })

        x2 = R((
            'x <- GetAssayData(d[["RNA"]], slot="counts");'
            'list(x@i, x@p, x@x, rownames(x), colnames(x))'
        ))
        x2 = [numpy2ri.rpy2py(x) for _, x in x2.items()]
        x2 = xa.DataArray(
            sparse.GCXS(
                (x2[2], x2[0], x2[1]), 
                shape=(len(x2[3]), len(x2[4])),
                compressed_axes=(1,)
            ),
            [('feature_id', x2[3]), ('cell_id', x2[4])],
            name='counts'
        )

        x5 = R((
            'x<-Embeddings(d[["umap"]]);'
            'list(x, rownames(x), colnames(x))'
        ))
        x5 = [numpy2ri.rpy2py(x) for _, x in x5.items()]
        x5 = xa.DataArray(
            x5[0],
            [('cell_id', x5[1]), ('umap_dim', x5[2])],
            name='umap'
        )

        x4 = xa.merge([x1, x2, x3, x5], join='inner')

        return x4
    
    def make_cache(self):
        _ = self.c2_wb_pbmc
        _ = self.c2_neutrophils_integrated

data = _data()

# %%
if __name__ == '__main__':
    self = data
     

# %%
