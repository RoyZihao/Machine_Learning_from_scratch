#### Marginal Likelihood calculations
#contour plot using ggplot2
library(data.table)
library(ggplot2)
library(reshape2) #for melt()
library(lattice) #for filled.contour()
library(RColorBrewer) #for brewer.pal()

## Define observed data
ss = matrix( c(20,40,
               2,4,
               20,80), byrow=TRUE, ncol=2)

ss = matrix( c(50,10000,
               160,20000,
               180,60000,
               0,100,
               0,5,
               1,2), byrow=TRUE, ncol=2)

colnames(ss) = c("x","n")

## Define data likelihood function and log-likelihood functions data
## likelihood function, and data log-likelihood function
marglik <- function(aa,bb) {
    prod(beta(ss[,"x"]+aa, ss[,"n"]-ss[,"x"]+bb)) / (beta(aa,bb)^nrow(ss))
}

margloglik <- function(aa,bb) {
    sum(lbeta(ss[,"x"]+aa, ss[,"n"]-ss[,"x"]+bb)) - nrow(ss) * lbeta(aa,bb)
}
## Define version that takes vector input for optimization
marglikv <- function(v) return(marglik(v[1],v[2]))
margloglikv <- function(v) return(margloglik(v[1],v[2]))

## Find maximum of log likelihood
control = list(fnscale=-50)  # optim find minumum, setting scale <=0  makes if finds max
opt=optim(par=c(5,5), margloglikv, lower = rep(0.01, 2), upper = rep(7000, 2), method="L-BFGS-B", control=control)
opt

opt$par  # empirical Bayes estimates of a and b
aEmpBayes = opt$par[1]
bEmpBayes = opt$par[2]
empBayesPriorMean = aEmpBayes / (aEmpBayes + bEmpBayes); empBayesPriorMean
empBayesPriorSD = sqrt( aEmpBayes * bEmpBayes / ((aEmpBayes+bEmpBayes)^2 * (aEmpBayes+bEmpBayes+1)) ); empBayesPriorSD

results = as.data.table(ss)
results[, mle := x/n]
results[, posteriorA := aEmpBayes + x]
results[, posteriorB := bEmpBayes + n - x]
results[, posteriorMean := posteriorA / (posteriorA + posteriorB)]
results[, posteriorSD := sqrt( posteriorA * posteriorB /((posteriorA+posteriorB)^2 * (posteriorA+posteriorB+1)) )]
results[, map := (posteriorA - 1) / (posteriorA + posteriorB - 2)]
### Add columns formatted the way I want for printing
results[, NumClicks := x]
results[, NumImpressions := n]
results[, MLE := paste0(round(100*mle,2),"%")]
results[, PosteriorMean := paste0(round(100*posteriorMean,2),"%")]
results[, PosteriorSD := paste0(round(100*posteriorSD,2),"%")]
results[, MAP := paste0(round(100*map,2),"%")]
## Generate latex code for table
library(xtable)
xtable(results[, list(NumClicks, NumImpressions, MLE, MAP, PosteriorMean, PosteriorSD)])

## Plot likelihood function
npts = 100
amax = opt$par[1]*2
bmax = opt$par[2]*2
a <- seq(amax/npts, amax,length.out=npts)
b <- seq(bmax/npts, bmax,length.out=npts)
d = CJ(a,b); setnames(d,c("V1","V2"),c("a","b"))
d[, sslik := marglik(a,b), by=rownames(d)]
d[, ssloglik := margloglik(a,b), by=rownames(d)]
v<-ggplot(d, aes(x=a,y=b,z=ssloglik))+
    geom_tile(aes(fill=ssloglik))+
    stat_contour(bins=6,aes(a,b,z=ssloglik), color="black", size=0.6)+
    scale_fill_gradientn(colours=brewer.pal(6,"YlOrRd"))
v
