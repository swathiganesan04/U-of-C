dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 1/Individual/"
data <- read.table(paste(dataPath,'Week1_Test_Sample.csv',sep = '/'), header=TRUE)

sample.u = data$u
sample.v = data$v

joint_distribution <- prop.table(table(data))
dimnames(joint_distribution)<-NULL

u_Marginal <- as.numeric(unlist(colSums(table(data)) / sum(table(data))))
v_Marginal <- as.numeric(unlist(rowSums(table(data)) / sum(table(data))))

u_Conditional_v <- unlist(list(c(0.35, 0.35, 0.30)))
v_Conditional_u <- unlist(list(c(0.10, 0.25, 0.35, 0.30)))

res <-list(Joint_distribution=joint_distribution,
           u_Marginal = u_Marginal,
           v_Marginal = v_Marginal,
           u_Conditional_v = u_Conditional_v,
           v_Conditional_u = v_Conditional_u          )

saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))
readRDS("/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 1/result.rds")
