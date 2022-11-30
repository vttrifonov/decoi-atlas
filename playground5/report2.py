# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground5'

# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ipywidgets import interact
from . import analysis, gmm_kl
from .._helpers import plot_table

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

# %% [markdown]
#  The goal of this note is to explore the possibility to clump single cell data using GMM,
#  and perform gene set analysis on (the centroid) of each clump. The clumps in this context should
#  be clusters of cells smaller than a cell type cluster, but bigger than just having just a few cells.
#  The idea is that a bigger clump will agregate reads from several similar cells and so gene set analysis
#  will be more robust. Once we do gene set analysis on the centroids we cluster (again using GMM) the
#  pathway based on their enrichment score across the centroids with the goal to provide a higher level
#  structure on the gene sets. We focus on GMM here mostly because we can express this type of clustering
#  within a neutral network (PyTorch, Tensorflow) and hence incorporate fitting the GMM model as part of
#  more complex models.
# 
#  Data is

# %%
analysis.clust3(20).data.sizes

# %% [markdown]
#  Before PCA we center/normalize along cell_id. PCA is

# %%
analysis.clust3(20).svd

# %% [markdown]
#  We do GMM with 20 clusters on svd.v

# %%
x1 = xa.merge([
    analysis.data[['cell_integrated_snn_res.0.3']],
    analysis.data.umap.to_dataset('umap_dim'),
    analysis.clust3(20).clust.rename('clust')
], join='inner')
x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype(str)
x1['clust'] = x1['clust'].astype(str)

# %% [markdown]
#  Cell type clustering provided by the paper

# %%
(
    ggplot(x1.to_dataframe())+
        aes('UMAP_1', 'UMAP_2', color='cell_integrated_snn_res.0.3')+
        geom_point(alpha=0.1)
)

# %% [markdown]
#  GMM-20 clustering

# %%
(
    ggplot(x1.to_dataframe())+
        aes('UMAP_1', 'UMAP_2', color='clust')+
        geom_point(alpha=0.1)
)

# %% [markdown]
#  Overlap between cell type and GMM-20. Data for contingency able is ordered descending by resid and
#  for each cluster we select cell with highest resid as representative. Then data is ordered by cell
#  and then resid within cell. Missing cells are added at the end
# 
# 
#  We can see that cell type 6 did not represent any clusters. Cell type 1 was broken across 5 clusters.
#  Cell type 2 is broken across 5 clusters. Cell type 4 is represented by cluster 11 and cluster 16 was
#  shared with cell type 1.
# 
#  The number 20 for the clusters was chosen by eye: we wanted to break down the cell types a bit but asking for
#  too many clusters produced clusters with too few elements.

# %%
plot_table1(x1[['cell_integrated_snn_res.0.3', 'clust']].to_dataframe())

# %% [markdown]
#  Number of cells per cluster in GMM-20

# %%
x1 = analysis.clust3(20).clust.to_dataframe()
x1.value_counts()

# %% [markdown]
#  Overlap between cell and GMM-10. Cell types 8, 4, 3 and 6 did not represent any clusters -- too few clusters.

# %%
x1 = xa.merge([
    analysis.data[['cell_integrated_snn_res.0.3']],
    analysis.clust3(10).clust.rename('clust')
], join='inner')
x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype(str)
x1['clust'] = x1['clust'].astype(str)
plot_table1(
    x1[['cell_integrated_snn_res.0.3', 'clust']].to_dataframe()
)

# %% [markdown]
#  Overlap between cell and GMM-30. Clusters 19, 24, 23, 27 had too few cells -- too many clusters

# %%
x1 = xa.merge([
    analysis.data[['cell_integrated_snn_res.0.3']],
    analysis.clust3(30).clust.rename('clust')
], join='inner')
x1['cell_integrated_snn_res.0.3'] = x1['cell_integrated_snn_res.0.3'].astype(str)
x1['clust'] = x1['clust'].astype(str)
plot_table1(
    x1[['cell_integrated_snn_res.0.3', 'clust']].to_dataframe()
)

# %% [markdown]
#  We did not optimize the number of clusters further: the number 20 seemed to provide a good
#  trade-off for breaking down the cell types while having the clusters not too small (>200 cells)
# 
#  GMM-20 produced 20 centroids in 100D PC-spaces. We convert those 100D back to feature space to obtain
#  the input data to the enrichment analysis on the centroids. Data is

# %%
analysis.clust3(20).means.rename(clust='cell_clust').sizes

# %% [markdown]
#  We took the following gene sets

# %%
x1 = analysis.clust3(20).enrich.data[['sig', 'sig_prefix']].to_dataframe()
x1.groupby('sig_prefix').size().sort_values()

# %% [markdown]
#  Pathway enrichment was done by performing a t-test between features in the gene set and outside.
#  This gives a coef, t-value, and p-value for the enrichment of a gene set in a particular GMM-20 cluster. Data is

# %%
analysis.clust3(20).enrich.data.rename(clust='cell_clust')

# %% [markdown]
#  To cluster the gene sets we used the enrichment coefficient (coef) data for cluster centroids.

# %%
analysis.clust3(20).enrich.clust1(30).data.rename(clust1='cell_clust').sizes

# %% [markdown]
# Explore the signatures associated with each cell cluster

# %%
x1 = analysis.clust3(20).enrich.clust1(30).data.rename(clust1='cell_clust')
x1 = x1.to_dataframe().reset_index()
x1['sig'] = x1.sig.str.replace('^[^_]*_', '', regex=True)

@interact(
    sig_prefix=[''] + list(x1.sig_prefix.drop_duplicates()), 
    cell_clust=[''] + list(np.sort(x1.cell_clust.drop_duplicates()).astype(str)),
    sig='',
    coef = ['ascending', 'descending'],
    rows=(0, 100, 20)
)
def show_x4(sig_prefix='', cell_clust='', sig='', coef='descending', rows=20):
    pd.set_option('display.max_rows', rows)
    x5 = x1
    if sig_prefix!='':
        x5 = x5[x5.sig_prefix==sig_prefix]
    if cell_clust!='':
        x5 = x5[x5.cell_clust==int(cell_clust)]
    if sig!='':
        x5 = x5[x5.sig.str.contains(sig, regex=True)]
    return x5.sort_values('coef', ascending=(coef=='ascending')).head(rows)

# %% [markdown]
# We perform PCA on the centroid enrichment data. PC1 vs PC2 is shown below. 
# The data in red is obtained by permuting PC1. Not much structure can be seen 
# in the first 2 PCs

# %%
x1 = analysis.clust3(20).enrich.clust1(30)
x3 = x1.svd
x3 = x3.u * x3.s
x3 = x3.sel(pc=x3.pc<2)
x3['pc'] = xa.DataArray(['x', 'y'], [('pc', [0,1])])[x3.pc]
x3 = x3.to_dataset('pc')
x3 = xa.merge([x3, x1.clust.rename('clust')])
x3 = x3.to_dataframe()
x3['clust'] = x3.clust.astype(str)

print(
    ggplot(
        x3
    )+
        aes('x', 'y')+
        geom_point(
            x3.assign(x=lambda d: np.random.permutation(d.x)),
            alpha=0.1, color='red'
        )+
        geom_point(alpha=0.1)+
        labs(x='PC1', y='PC2')
)

# %% [markdown]
#  PCA of the enrichment scores shows that they are more ellipsoidal than expected (orange are singular values from permutated data)
#  but there are no signs of clustering (large gaps in the singular values)

# %%
x4 = analysis.clust3(20).enrich.clust1(30).svd.rename(clust1='cell_clust')
x4['cell_clust'] = x4.cell_clust.astype('str')[x4.cell_clust]

#pipe(lambda x: x**2/(x**2).sum()).\
x4['s'].query(pc='pc<19').to_series().\
    pipe(np.log2).\
    plot(style='.')

x4['rand_s'].query(pc='pc<19').to_series().\
    pipe(np.log2).\
    plot(style='.')

# %%
x4['v'].sel(pc=0).to_series().sort_values().plot()

# %%
x4['u'].sel(pc=0).to_series().sort_values().plot()

# %% [markdown]
# We performed GMM on the PCA of the enrichment scores. Picking the number of 
# clusters was again done by eye: we wanted  many clusters of gene sets but when 
# the number of clusters is too high the size of the clusters drops. In the 
# following analysis we used GMM-30 for clustering the enrichment 
# score.

# %%
analysis.clust3(20).enrich.clust1(30).clust.rename('sig_clust').to_dataframe().\
    groupby('sig_clust').size().sort_values()

# %%
x1 = analysis.clust3(20)
x2 = x1.enrich.clust1(30)
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
print(
    ggplot(x6)+
        aes('pc', 'np.log2(s)')+
        geom_point(aes(group='clust', color='clust'))
)

# %%
x1 = analysis.clust3(20)
x2 = x1.enrich.clust1(30)
x5 = x2.gmm.rename(clust='sig_clust')
#x5['sig_clust'] = x5.sig_clust.astype('str')[x5.sig_clust]
x4 = x2.svd.rename(clust1='cell_clust')
#x4['cell_clust'] = x4.cell_clust.astype('str')[x4.cell_clust]
x6 = (x5.means@x4.v)*x4.scale + x4['mean']
x6 = x6.rename('means').to_dataframe().reset_index()

x7 = xa.merge([
    analysis.data[['cell_integrated_snn_res.0.3']],
    x1.clust.rename('cell_clust')
], join='inner')
x7 = sm.stats.Table.from_data(x7.to_dataframe())
x7 = x7.resid_pearson.melt(ignore_index=False).reset_index()
x7 = x7.sort_values('value', ascending=False).\
    drop_duplicates(x7.columns[1]).\
    sort_values([x7.columns[0], 'value']).cell_clust

from scipy.cluster.hierarchy import dendrogram, linkage
x8 = x6.pivot_table(index='sig_clust', columns='cell_clust', values='means')
x8 = linkage(x8, method='average')
x8 = dendrogram(x8, no_plot=True)['leaves']

x6['cell_clust'] = x6.cell_clust.astype(pd.CategoricalDtype(x7))
x6['sig_clust'] = x6.sig_clust.astype(pd.CategoricalDtype(x8))

print(
    ggplot(x6)+
        aes('cell_clust', 'sig_clust')+
        geom_tile(aes(fill='means'))+
        scale_fill_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)

# %%
x1 = analysis.clust3(20).enrich.clust1(30)
x3 = x1.svd
x3 = x3.u * x3.s
x3 = x3.sel(pc=x3.pc<2)
x3['pc'] = xa.DataArray(['x', 'y'], [('pc', [0,1])])[x3.pc]
x3 = x3.to_dataset('pc')
x3 = xa.merge([x3, x1.clust.rename('clust')])
x3 = x3.to_dataframe()
x3['clust'] = x3.clust.astype(str)

print(
    ggplot(
        x3
    )+
        aes('x', 'y', color='clust')+
        geom_point(alpha=0.5)+
        labs(x='PC1', y='PC2')
)

# %% [markdown]
# Explore the gene sets represented by each signature cluster

# %%
x1 = analysis.clust3(20)
x2 = x1.enrich.clust1(30)
x3 = xa.merge([
    analysis.data.drop_dims('feature_id'),
    x1.enrich.sigs.drop('sig_prefix'),
    x1.enrich.means.rename('clust_means').rename(clust='cell_clust_id'),
    x1.clust.rename('cell_clust'),
    x1.gmm.pred.rename('cell_proba').rename(clust='cell_clust_id'),
    x2.data.sig_prefix.reset_coords(),
    x2.clust.rename('sig_clust'),
    x2.gmm.pred.rename('sig_proba').rename(clust='sig_clust_id'),
    x2.means.rename(clust='sig_clust_id', clust1='cell_clust_id').rename('sig_clust_means')
], join='inner')
#x4 = x3.log1p_rpk
#x3['z_log1p_rpk'] = (x4-x4.mean(dim='cell_id'))/x4.std(dim='cell_id')
x3 = xa.merge([
    x3.sig_prefix,
    x3.drop_dims('umap_dim'),
    x3.umap.to_dataset(dim='umap_dim')
])
x3['sig'] = x3.sig.str.replace('^[^_]*_', '')

# %%
x4 = xa.merge([
    x3.sig_prefix,
    x3.sig_proba.rename(sig_clust_id='sig_clust')
])
x4 = x4.to_dataframe().reset_index()

@interact(
    sig_prefix=[''] + list(x4.sig_prefix.drop_duplicates()),
    sig_clust=[''] + list(x4.sig_clust.drop_duplicates().astype(str)),
    sig='',
    rows=(0, 100, 20)
)
def show_x4(
    sig_prefix='', 
    sig_clust='', 
    sig='',
    rows=20
):
    pd.set_option('display.max_rows', rows)
    x5 = x4
    if sig_prefix!='':
        x5 = x5[x5.sig_prefix==sig_prefix]
    if sig_clust!='':
        x5 = x5[x5.sig_clust==int(sig_clust)]
    if sig!='':
        x5 = x5[x5.sig.str.contains(sig, regex=True)]
    return x5.sort_values('sig_proba', ascending=False).head(rows)

# %%
sig = 'INTERFERON_GAMMA_RESPONSE'
x4 = x3
x4 = x4.sel(sig=sig)
x4['set'] = xa.apply_ufunc(
    np.matmul, x4.feature_entrez, x4.set.todense(), 
    input_core_dims=[['feature_id', 'gene'], ['gene']],
    output_core_dims=[['feature_id']]
)>0
x4 = x4[['set', 'clust_means']].rename(cell_clust_id='cell_clust').to_dataframe().reset_index()
x4['cell_clust'] = x4.cell_clust.astype(str)

# %%
(
    ggplot(x4)+aes('cell_clust', 'clust_means', color='set')+
        geom_boxplot()
)



