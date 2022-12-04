# %%
if __name__ == '__main__':
    __package__ = 'decoi_atlas.playground5.report2'

# %%
import numpy as np
import pandas as pd
import xarray as xa
import ipywidgets as widgets
from IPython.display import display, clear_output
import statsmodels.api as sm
from plotnine import *
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from .. import _analysis, analysis
from ...common.caching import compose, lazy
from ...reactive import Output, VBox, react

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

    ctrls = VBox(dict(
        filter=VBox(dict(
            sig_prefix = widgets.Dropdown(value='', description='sig_prefix', options=[''] + list(x4.sig_prefix.drop_duplicates())),
            sig_clust = widgets.Dropdown(value='', description='sig_clust', options=[''] + list(x4.sig_clust.drop_duplicates().astype(str))),
            sig = widgets.Text(value='', description='sig'),
            sig_size = widgets.IntRangeSlider(value=[10, 500], description='sig_size', min=1, max=x4.sig_size.max()),
            sig_proba = widgets.FloatSlider(value=0.9, description='sig_proba', min=0, max=1, step=0.1)
        )),
        pager=VBox(dict(
            rows = widgets.IntSlider(value=20, description='rows', min=20, max=100, step=20),
            page = widgets.IntSlider(value=1, description='page', min=1, max=1, step=1)            
        )),
        label = widgets.Label(),
        out = Output()
    ))

    @react(*ctrls.c.filter.children)
    def x5(sig_prefix, sig_clust, sig, sig_size, sig_proba):
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

        x5 = x5.sort_values('sig_proba', ascending=False)
        return x5

    @react(x5, *ctrls.c.pager.children)
    def x6(x, r, p):
        n = x.shape[0]

        ctrls.c.label.value = f'{n} rows.'  
        page = ctrls.c.pager.c.page 
        if n==0:
            page.min = 0
            page.max = 0
        else:
            page.max = np.ceil(n/r)
            page.min = 1         

        p = min(page.max, p)
        x = x.iloc[(p-1)*r:p*r]
        pd.set_option('display.max_rows', r)
        return x

    ctrls.c.out.react(x6)

    return ctrls

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
if __name__ == '__main__':
    self = analysis.clust3(20).enrich.clust1(30)

    # %%
    self.sigs_for_clust

# %%
