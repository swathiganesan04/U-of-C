
dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 3/Individual/"
dat <- read.table(paste0(dataPath, 'Week3_Test_Sample.csv'), header=TRUE)

mean <- dat$x[1]
stdv <- dat$x[2] 
intensity <- dat$x[3]

sample <- dat$x[4:503]

datNorm <- qnorm(sample, mean=mean, sd=stdv)
hist(datNorm)

datExp <- qexp(sample, rate=intensity, lower.tail = TRUE, log.p = FALSE)
hist(datExp)

res<-cbind(datNorm=datNorm,datExp=datExp)
write.table(res, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)
