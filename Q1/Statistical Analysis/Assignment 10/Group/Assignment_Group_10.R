dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 10/Group/"
dta<-read.csv(paste(dataPath,'Country-data.csv',sep='/'),row.names=1)

dtaNorm<-apply(dta,2,function(z) z/sd(z))
dtaNorm<-as.data.frame(dtaNorm)

pca <- princomp(dtaNorm)
summary(pca)
M <- 4

pca1$loadings

res<-list( factorsNumber = M,
           extremes = extremes,
           selectedFactor = selectedFactor,
           selectedCountries =  selectedCountries,
           totalAllocations = quotas,
           targetChanges =  targetChanges
)
saveRDS(res,paste(dataPath,'result.rds',sep='/'))
