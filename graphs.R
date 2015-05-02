data1 <- read.csv("output_qlearning.csv", header=F)
data2 <- read.csv("output_qlearning_init_2.csv", header=F)
df1 <- data.frame(Epoch=1:100, Score=data1$V1, Initialization=rep("Naive",100))
df2 <- data.frame(Epoch=1:100, Score=data2$V1, Initialization=rep("Support",100))
df <- rbind(df1,df2)
require("ggplot2")

p <- ggplot(df, aes(x=Epoch,y=Score,color=Initialization))
p + geom_point() + ggtitle("Scores of different initializations") + ylab("Score") + xlab("Epoch") + theme_bw()