# %%
if __name__ == '__main__':
    __file__ = os.path.expanduser('~/projects/empty.py')
    __package__ = 'decoi_atlas.playground5.report2'

# %%
import numpy as np
import pandas as pd
import xarray as xa
from plotnine import *
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import plotly.express as px
from . import analysis

# %%


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
#  Before PCA we center/normalize along cell_id. We do GMM with 20 clusters on svd.v from PCA

# %% [markdown]
#  Cell type clustering provided by the paper

# %%
px.scatter(
    analysis.cell_type_clusters, 
    x='UMAP_1', y='UMAP_2', color='cell_integrated_snn_res.0.3'
)

# %% [markdown]
#  GMM-20 clustering

# %%
px.scatter(
    analysis.clust3(20).cell_type_clusters,
    x='UMAP_1', y='UMAP_2', color='clust'
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
#  too many clusters produced clusters with too few elements.  On the other hand, with few clusters some 
#  cell types do not get represented

# %%
analysis.cluster_to_cell_overlap

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
analysis.clust3(20).enrich.data[['sig', 'sig_prefix']].to_dataframe().\
    groupby('sig_prefix').size().sort_values()

# %% [markdown]
#  Pathway enrichment was done by performing a t-test between features in the gene set and outside.
#  This gives a coef, t-value, and p-value for the enrichment of a gene set in a particular GMM-20 cluster. 
#  
#  To cluster the gene sets we used the enrichment coefficient (coef) data for cluster centroids.

# %%
analysis.clust3(20).enrich.clust1(30).data.rename(clust1='cell_clust').sizes

# %% [markdown]
# Explore the signatures associated with each cell cluster

# %%
analysis.clust3(20).enrich.clust1(30).cell_cluster_sigs

# %% [markdown]
# We perform PCA on the centroid enrichment data. PC1 vs PC2 is shown below. 
# The data in red is obtained by permuting PC1. Not much structure can be seen 
# in the first 2 PCs

# %%
x3 = analysis.clust3(20).enrich.clust1(30).pca1
(
    ggplot(x3)+
        aes('x', 'y')+
        geom_point(
            x3.assign(x=lambda d: np.random.permutation(d.x)),
            alpha=0.1, color='red'
        )+
        geom_point(alpha=0.1)+
        labs(x='PC1', y='PC2')+
        theme(figure_size=(3,3))
)

# %% [markdown]
#  PCA of the enrichment scores shows that they are more ellipsoidal than expected (orange are singular values from permutated data)
#  but there are no signs of clustering (large gaps in the singular values)

# %%
analysis.clust3(20).enrich.clust1(30).pca2

# %% [markdown]
# We performed GMM on the PCA of the enrichment scores. Picking the number of 
# clusters was again done by eye: we wanted  many clusters of gene sets but when 
# the number of clusters is too high the size of the clusters drops. In the 
# following analysis we used GMM-30 for clustering the enrichment 
# score.

# %% [markdown]
# Cluster sizes for GMM-30 on enrichment scores

# %%
(
    ggplot(analysis.clust3(20).enrich.clust1(30).clust_sizes)+
        aes('sig_clust', 'num_sigs')+
        geom_bar(stat='identity')+scale_y_log10()+
        theme(figure_size=(6, 3))
)

# %% [markdown]
# Cluster for GMM-30 on enrichment scores on PC1 vs PC2

# %%
px.scatter(
    analysis.clust3(20).enrich.clust1(30).pca3, 
    x='x', y='y', color='clust', hover_name='sig'
)

# %% [markdown]
# Signature cluster covariance eigenvalues

# %%
(
    ggplot(analysis.clust3(20).enrich.clust1(30).clust_covs)+
        aes('pc', 'np.log2(s)')+
        geom_point(aes(group='clust', color='clust'))+
        theme(figure_size=(3,3))
)

# %% [markdown]
# Signature cluster means

# %%
(
    ggplot(analysis.clust3(20).enrich.clust1(30).clust_means)+
        aes('cell_clust', 'sig_clust')+
        geom_tile(aes(fill='means'))+
        scale_fill_gradient2(
            low='blue', mid='white', high='red',
            midpoint=0
        )
)

# %% [markdown]
# Explore the gene sets represented by each signature cluster

# %%
analysis.clust3(20).enrich.clust1(30).sigs_for_clust

# %% [markdown]
# Explore gene set across cell clusters

# %%
analysis.clust3(20).enrich.clust1(30).expr_for_clust


