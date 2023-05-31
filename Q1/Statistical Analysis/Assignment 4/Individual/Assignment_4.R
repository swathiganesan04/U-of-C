dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 4/Individual/"
dat <- read.table(paste(dataPath,'Week4_Test_Sample.csv',sep = '/'), header=TRUE)

print(dat)
plot(dat$X,dat$Y)

Estimated.LinearModel <- lm(Y ~ X,data=dat)
names(Estimated.LinearModel)

Estimated.LinearModel$coefficients
plot(Estimated.LinearModel$residuals)
Estimated.LinearModel$fitted.values


Estimated.Residuals <- lm(Y ~ X,data=dat)
Estimated.Residuals <- residuals(Estimated.Residuals)

# Isolate positive values
pos.values <- Estimated.Residuals > 0

# Returns values that are TRUE (positive values)
true.values <- Estimated.Residuals[Estimated.Residuals > 0]

# Returns mean of values that are TRUE
mean.values <- mean(Estimated.Residuals[Estimated.Residuals > 0])

# Return value based on logic
unscrambled.selection.sequence <- ifelse(Estimated.Residuals > 0, 1, 0)
unscrambled.selection.sequence

# Save File
res <- list(Unscrambled.Selection.Sequence =  unscrambled.selection.sequence)
write.table(res, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)
