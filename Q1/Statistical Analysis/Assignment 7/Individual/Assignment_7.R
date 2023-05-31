# Assignment 7
dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 7/Individual/"

# SECTION 1.1 - READING THE DATA
test_dat <- read.table(paste(dataPath,'Week7_Test_Sample.csv',sep = '/'), header=TRUE)
head(test_dat)

# SECTION 1.2 - Fit linear models using: no inputs, only Input1, only Input2, both Input1 and Input2.
fit.1<-lm(Output~1,data=test_dat) #output y
fit.1.2<-lm(Output~1+Input1,data=test_dat) #first predictor values 
fit.1.3<-lm(Output~1+Input2,data=test_dat) #second predictor values
fit.1.2.3<-lm(Output~.,data=test_dat)

# SECTION 2.1 - OUPTUTS OF ANOVA()
anova(fit.1.2)$"Sum Sq" #= 305.8921  202.3953 #Q1 & Q2
anova(fit.1.2)$"Pr(>F)"[1] #= 1.217524e-101
anova(fit.1.2)

anova(fit.1.3)$"Sum Sq" #= 0.9462866  507.3410508 #Q3 & Q4
anova(fit.1.3)

c(anova(fit.1,fit.1.2.3)$F[2],summary(fit.1.2.3)$fstatistic[1]) #= 526.301
#378.5523 #Q5
summary(fit.1.2.3)

anova(fit.1.2.3)$"Pr(>F)"[1] # 9.45874e-102 $Q6
c(anova(fit.1,fit.1.2.3)$F[2],summary(fit.1.2.3)) 
anova(fit.1,fit.1.2.3)

#Q7 Second

