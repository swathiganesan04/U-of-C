

dataPath <- "/Users/swathiganesan/Downloads/"
dat <- read.table(paste(dataPath,'Week5_Test_Sample.csv',sep = '/'), header=TRUE)
head(dat)
plot(dat$Input,dat$Output, type="p",pch=19)
nSample<-length(dat$Input)

#Plot  
plot(dat$Input,(dat$Output-mean(dat$Output))^2, type="p",pch=19,
     ylab="Squared Deviations")


#Linear Model Fit
GeneralModel <- lm(dat$Output ~ dat$Input, data = dat)

clusteringparabola <- (GeneralModel$fitted.values - mean(dat$Output))^2


plot(dat$Input,(dat$Output-mean(dat$Output))^2, type="p",pch=19,
     ylab="Squared Deviations")
points(dat$Input,clusteringparabola,pch=19,col="red")


Unscrambling.Sequence.Steeper.var <- ((dat$Output-mean(dat$Output))^2>clusteringparabola)
head(Unscrambling.Sequence.Steeper.var,30)

#Separatating sample into steeper and flatter variables
Subsample.Steeper.var<-
  data.frame(steeperInput.var=dat$Input,steeperOutput.var=rep(NA,nSample))
Subsample.Flatter.var<-
  data.frame(flatterInput.var=dat$Input,flatterOutput.var=rep(NA,nSample))



#Fill in the unscrambled outputs instead of NULL values where necessary
Subsample.Steeper.var[Unscrambling.Sequence.Steeper.var,2]<-
  dat[Unscrambling.Sequence.Steeper.var,1]
Subsample.Flatter.var[!Unscrambling.Sequence.Steeper.var,2]<-
  dat[!Unscrambling.Sequence.Steeper.var,1]


# Check the first 10 rows (head)
head(cbind(dat,Subsample.Steeper.var,Subsample.Flatter.var),10)

#Plotting clusters of the variance data and the separating parabola 
?points

plot(dat$Input,
     (dat$Output-mean(dat$Output))^2,
     type="p",pch=19,ylab="Squared Deviations")
points(dat$Input,clusteringparabola,pch=19,col="red")
points(dat$Input[Unscrambling.Sequence.Steeper.var],
       (dat$Output[Unscrambling.Sequence.Steeper.var]-
          mean(dat$Output))^2,
       pch=19,col="blue")
points(dat$Input[!Unscrambling.Sequence.Steeper.var],
       (dat$Output[!Unscrambling.Sequence.Steeper.var]-
          mean(dat$Output))^2,
       pch=19,col="green")

#Extracting points from plot
x1<-Subsample.Steeper.var$steeperInput.var
y1<-Subsample.Steeper.var$steeperOutput.var
x2<-Subsample.Flatter.var$flatterInput.var
y2<-Subsample.Flatter.var$flatterOutput.var



#Fitting the new linear model with corrected variable names
GeneralModel <- lm(dat$Output ~ dat$Input, data = dat)
mSteep <- lm(y1 ~ x1, data=Subsample.Steeper.var) 
mFlat <-  lm(y2 ~ x2, data=Subsample.Flatter.var) 

#Saving File
res <- list( GeneralModel = GeneralModel,mSteep = mSteep,mFlat = mFlat)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))



