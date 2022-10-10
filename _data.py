#%%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

# %%
import xarray as xa
from ._helpers import config
from .common.caching import compose, lazy, XArrayCache

# %%
class _data:
    storage = config.cache / 'data'

    @compose(property, lazy, XArrayCache())
    def c2_wb_pbmc(self):
        from rpy2.robjects import r as R
        from rpy2.robjects import pandas2ri, numpy2ri
        import sparse

        R(f'source("{config.root}/_data.R")')

        x1 = R('data$c2_wb_pbmc@meta.data')
        x1 = pandas2ri.rpy2py(x1)
        x1.index.names = ['cell_id']
        x1 = x1.to_xarray()
        x1 = x1.rename({
            k: 'cell_'+k for k in x1.keys()
        })

        x3 = R('data$c2_wb_pbmc[["RNA"]]@meta.features')
        x3 = pandas2ri.rpy2py(x3)
        x3.index.names = ['feature_id']
        x3 = x3.to_xarray()
        x3 = x3.rename({
            k: 'feature_'+k for k in x3.keys()
        })

        x2 = R((
            'x <- GetAssayData(data$c2_wb_pbmc[["RNA"]], slot="counts");'
            'list(x@i, x@p, x@x, rownames(x), colnames(x))'
        ))
        x2 = [numpy2ri.rpy2py(x) for _, x in x2.items()]
        x2 = xa.DataArray(
            sparse.GCXS((x2[2], x2[0], x2[1]), shape=(len(x2[3]), len(x2[4]))),
            [('feature_id', x2[3]), ('cell_id', x2[4])],
            name='counts'
        )

        x4 = xa.merge([
            x2.to_dataset(), x1, x3
        ], join='inner')

        return x4


data = _data()

# %%
if __name__ == '__main__':
    self = data
     
    # %%
    self.c2_wb_pbmc

# %%
