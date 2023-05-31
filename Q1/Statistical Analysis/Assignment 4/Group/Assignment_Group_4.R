library(fitdistrplus)

dataPath <- "/Users/swathiganesan/Downloads/"
data<-read.csv(file=paste(dataPath,"Method_Moments_Data.csv",sep="/"))


mean.a <- round(mean(data$A),2)
sd.a <- round(sd(data$A),2)

lambda.a<-round(1/mean.a,2)
Exponential.fit<-fitdistr(data$B,"normal")
c(Exponential.fit$estimate,sd=Exponential.fit$sd)
ks.test(data$A,"pnorm")

Exponential.fit<-fitdistr(data$A,"exponential")
c(Exponential.fit$estimate,sd=Exponential.fit$sd)

mean.b <- round(mean(data$B),2)
sd.b <- round(sd(data$B),2)
ks.test(data$A, data$B)

lambda.b<-round(1/mean.b,2)
Exponential.fit<-fitdistr(data$B,"normal")
c(Exponential.fit$estimate,sd=Exponential.fit$sd)
ks.test(data$B,"pnorm",lambda.a)


LM <- lm(Y ~ X,data=data)
LM
summary(LM)
Y_bar<-rep(mean(data$Y),500)
X_X_bar<-data$X-mean(data$X)

#doing dot product
Y_bar%*%X_X_bar
#dot product is 0 hence orthogonal

sd.resid <- sd(LM$residuals)
sd.resid



