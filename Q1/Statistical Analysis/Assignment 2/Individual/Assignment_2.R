
dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 2/Individual/"
df <- read.table(paste0(dataPath, 'Week2_Test_Sample.csv'), header=TRUE)

# Standard deviation
sdX <- round(sd(df$x), digits = 2)
sdY <- round(sd(df$y), digits = 2)

# Correlation co-efficient
cXY <- round(cor(df$x, df$y), digits = 2)


# Slope 
a<-((sdY/sdX)*cXY)

result <- data.frame(sdX=sdX, sdY=sdY, cXY=cXY, a=a)  
write.table(result, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)

