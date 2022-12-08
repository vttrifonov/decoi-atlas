import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri
from rpy2.robjects import r as R
import xarray as xa
import numpy as np

dict2ri = ro.conversion.Converter('dict converter')
@dict2ri.py2rpy.register(dict)    
def _dict_py2rpy(x):
    return ro.ListVector(x)
@dict2ri.rpy2py.register(ro.ListVector)
def _dict_rpy2py(x):
    co = ro.conversion.get_conversion()
    return {k: co.rpy2py(v) for k, v in x.items()}
    
xa2ri = ro.conversion.Converter('xarray converter')
@xa2ri.py2rpy.register(xa.DataArray)    
def _xa_py2rpy(d):
    with localconverter(ro.default_converter+numpy2ri.converter) as co:
        array = co.py2rpy(np.asarray(d.data, order='C'))
        dimnames = ro.ListVector({k: d[k].data for k in d.dims})
        dims = co.py2rpy(np.asarray(d.data.shape))
    array = R.array(array, dim=dims)
    array.dimnames = dimnames
    return array
@xa2ri.rpy2py.register(ro.vectors.Array)
def _xa_rpy2py(x):
    with localconverter(ro.default_converter+numpy2ri.converter) as co:
        dimnames = co.rpy2py(x.dimnames)
        matrix = co.rpy2py(x)
    return xa.DataArray(matrix, coords=dimnames)

    