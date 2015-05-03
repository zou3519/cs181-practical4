fname = file.choose()
data1 = read.table(fname, header=FALSE, sep=",")
data1

df1 <- data.frame(Epoch=1:751, Score=data1$V1)

require("ggplot2")

p <- ggplot(df1, aes(x=Epoch,y=Score))
p + geom_point() + ggtitle("Score for Model-Based Approach") + ylab("Score") + xlab("Epoch") + theme_bw()

