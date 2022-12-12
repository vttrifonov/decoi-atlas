library(limma)
library(data.table)

fit <- function(d, s, f) {
    v <- voom(d, design=model.matrix(as.formula(f), data=s))
    l <- lmFit(v)
    l <- eBayes(l)
    r <- list(voom=v$E, wts=v$weights, coef=l$coef, t=l$t, p=l$p.value)
    dimnames(r$wts)<-dimnames(r$voom)
    l <- dimnames(r$coef)
    names(l) <- c(names(dimnames(d))[1], 'var')
    for(n in c('coef', 't', 'p'))
        dimnames(r[[n]]) <- l
    r
}

fit1 <- function(d, s, f) {
    l <- lmFit(d, design=model.matrix(as.formula(f), data=s))
    l <- eBayes(l)
    r <- list(coef=l$coef, t=l$t, p=l$p.value)
    l <- dimnames(r$coef)
    names(l) <- c(names(dimnames(d))[1], 'var')
    for(n in c('coef', 't', 'p'))
        dimnames(r[[n]]) <- l
    r
}
