library(Seurat)
library(magrittr)
library(data.table)
library(magrittr)
source('common/module.R')
load.module('common')
source('_helpers.R')

data <- obj() %>%
    lazy_prop(c2_wb_pbmc, {
        UpdateSeuratObject(readRDS(file.path(
            config$cache, 'download',
            'seurat_COVID19_freshWB-PBMC_cohort2_rhapsody_jonas_FG_2020-08-18.rds'
        )))
    }) %>%

    lazy_prop(c2_neutrophils_integrated, {
        UpdateSeuratObject(readRDS(file.path(
            config$cache, 'download',
            'seurat_COVID19_Neutrophils_cohort2_rhapsody_jonas_FG_2020-08-18.rds'
        )))
    })