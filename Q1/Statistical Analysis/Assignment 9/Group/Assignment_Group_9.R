dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 9/Group/"
data <- read.csv(paste(dataPath, "test_sample 3.csv", sep="/"), stringsAsFactors=T)

levels(data$t)
A<-data$t
contrasts(A)

summary(lm(aov(data$x~A)))

TukeyHSD(aov(x~t,data))

summary(lm(aov(x~t,data = data)))


N <- length(data$x) # total sample size
k <- length(unique(data$t)) # number of treatments
n <- length(data$x) / k # number of samples per group (since sizes are equal)

# Mean Square
groups <- split(data, data$t)

sse <- sum(Reduce('+', lapply(groups, function(x) {
  (length(x[,1]) - 1) * sd(x[,1])^2
})))

mse <- sse / (N - k)
mse
q.value <- qtukey(p = 0.95, nmeans = k, df = N - k)
q.value

# Tukey Honestly Significant Difference
tukey.hsd <- q.value * sqrt(mse / n)
tukey.hsd

means <- tapply(data$x, data$t, mean)
B.A.diff <- means[2] - means[1] #
C.A.diff <- means[3] - means[1]
D.A.diff <- means[4] - means[1] 
E.A.diff <- means[5] - means[1] #
F.A.diff <- means[6] - means[1] #

C.B.diff <- means[3] - means[2]
D.B.diff <- means[4] - means[2]
E.B.diff <- means[5] - means[2] #
F.B.diff <- means[6] - means[2] #
D.C.diff <- means[4] - means[3] #
E.C.diff <- means[5] - means[3]
F.C.diff <- means[6] - means[3]
E.D.diff <- means[5] - means[4]
F.D.diff <- means[6] - means[4]
F.E.diff <- means[6] - means[5] #

#1 2 3 4 5 6
#A B C D E F
for (i in list(B.A.diff, C.A.diff, D.A.diff, E.A.diff, F.A.diff, C.B.diff,D.B.diff,E.B.diff,F.B.diff,
               D.C.diff,E.C.diff,F.C.diff,E.D.diff,F.D.diff,F.E.diff)) {
  print(abs(i) >= tukey.hsd)
}

#AB,AE,AF,BE,BF,CD,EF