datapath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 8/Group/"


test_data <-read.csv(file=paste(datapath,"test_sample 2.csv",sep="/"),
                     row.names=1,header=TRUE,sep=",")

Window.width<-20; Window.shift<-5

all.means<-rollapply(test_data,width=Window.width,by=Window.shift,by.column=TRUE, mean)
head(all.means,10)

lagdiff<-diff(as.matrix(test_data))
rolling.sd<-rollapply(lagdiff,width=Window.width,by=Window.shift,by.column=TRUE, sd)
rolling.dates<-rollapply(test_data[-1,],width=Window.width,by=Window.shift,
                         by.column=FALSE,FUN=function(z) rownames(z))
rownames(rolling.sd)<-rolling.dates[,10]

high.volatility.periods<-rownames(rolling.sd)[rolling.sd[,8]>.3]
high.volatility.periods




Coefficients<-rollapply(test_data,width=Window.width,by=Window.shift,by.column=FALSE,
                        FUN=function(z) coef(lm(Output1~USGG3M+USGG5YR+USGG30YR,data=as.data.frame(z))))
rolling.dates<-rollapply(test_data[,1:8],width=Window.width,by=Window.shift,by.column=FALSE,
                         FUN=function(z) rownames(z))

rownames(Coefficients)<-rolling.dates[,10]
Coefficients[1:10,]
dim(Coefficients)
Coefficients[,4]
high.slopeSpread.periods<-rownames(Coefficients)[abs(Coefficients[,3]-Coefficients[,4])>3]
high.slope5Y<-rownames(Coefficients)[Coefficients[,3]>2.5]

high.slopeSpread.periods
high.slope5Y

# R-squared
r.squared<-rollapply(test_data,width=Window.width,by=Window.shift,by.column=FALSE,
                     FUN=function(z) summary(lm(Output1~USGG3M+USGG5YR+USGG30YR,data=as.data.frame(z)))$r.squared)
r.squared<-cbind(rolling.dates[,10],r.squared)
r.squared[1:10,]


low.r.squared<-r.squared[r.squared[,2]<.9,1]
low.r.squared

#P-values
Pvalues<-rollapply(test_data,width=Window.width,by=Window.shift,by.column=FALSE,
                   FUN=function(z) summary(lm(Output1~USGG3M+USGG5YR+USGG30YR,data=as.data.frame(z)))$coefficients[,4])
rownames(Pvalues)<-rolling.dates[,10]
Pvalues[1:10,]

USGG3M_insignificant<-rownames(Pvalues)[Pvalues[,2]>.05]
USGG5Y_insignificant<-rownames(Pvalues)[Pvalues[,3]>.05]
USGG30Y_insignificant<-rownames(Pvalues)[Pvalues[,4]>.05]


res <- list(high.volatility.periods=high.volatility.periods,
            high.slopeSpread.periods=high.slopeSpread.periods,
            high.slope5Y=high.slope5Y,
            low.r.squared = low.r.squared,
            USGG3M_insignificant=USGG3M_insignificant,
            USGG5Y_insignificant=USGG5Y_insignificant,
            USGG30Y_insignificant=USGG30Y_insignificant
            )

saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))
 