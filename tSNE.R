#########################
#This applies t-SNE to the Allstate data 
#input: Uses the cleaned/processed claims data saved from the xgboost model
#output: t-SNE features for the claims data
#

library(methods)
library(Rtsne)
library(data.table)

set.seed(2016)
load(file="~/Desktop/kaggle/AllState/input/11152016-ProcessedTrainTest.RData")

y=train[,loss]
train[,loss:=NULL]

test_id<-test[,id]
train_id<-train[,id]
train[,id:=NULL]
test[,id:=NULL]

#Remove combination features for speed
train<-train[ , !grepl( "_" , names( train) ),with=FALSE ]
test<-test[ , !grepl( "_" , names( test) ),with=FALSE]

x = rbind(train,test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))

tsne <- Rtsne(as.matrix(x), check_duplicates = FALSE, pca = FALSE, 
              perplexity=10, theta=0.5, dims=2)

plot(tsne$Y, asp = 1, pch = 20, col = "blue", 
     cex = 0.75, cex.axis = 1.25, cex.lab = 1.25, cex.main = 1.5, 
     xlab = "t-SNE dimension 1", ylab = "t-SNE dimension 2", 
     main = "2D t-SNE projection")


#Seperate out data
tsne_feature<-tsne$Y
tsne_train<-tsne_feature[1:nrow(train),]
tsne_test<-tsne_feature[(nrow(train)+1):nrow(tsne_feature),]

save(tsne,tsne_train,tsne_test,file="Dec7-tSNEFeaturesNov15Data_10Perplex_NoCombo.Rdata")
