library(limma)

fit <- function(d, s, f) {
    v <- voom(d, design=model.matrix(as.formula(f), data=s))
    l <- eBayes(lmFit(v))
    r <- list(voom=v$E, coef=l$coef, t=l$t, p=l$p.value)
    l <- dimnames(r$coef)
    names(l) <- c(names(dimnames(d))[1], 'var')
    for(n in c('coef', 't', 'p'))
        dimnames(r[[n]]) <- l
    r
}
