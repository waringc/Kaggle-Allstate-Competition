#########################
#Calculates k-nearest neighbour for Allstate data and
#provides test and out of fold train predictions
#input: Uses the cleaned/processed claims data saved from the xgboost model
#output: kNN prediction of claims
#


library(data.table)
library(dplyr)
library(Matrix)
library(Metrics)
library(forecast)
library(FNN)

load(file="~/Desktop/kaggle/AllState/input/11152016-ProcessedTrainTest.Rdata")
#feat<-c("cont1","cont2","cont3","cont4", "cont5","cont6","cont7","cont8","cont9","cont10","cont11","cont12","cont13","cont14" )

y<- train$loss
y<-exp(y)-200

train[,loss:=NULL]
train[,id:=NULL]
test[,id:=NULL]

#train=train[,names(train)%in%feat, with= FALSE]
train=train[,1:131, with= FALSE]
test=test[,1:131, with= FALSE]

k=2048
#Do OOF Predictions
results_cv <- knn.reg(train = train, y = y, k = k)
err= mae(results_cv$pred,y)
cat(paste0("Error:", err,"\n"))

#Train on test
results <- knn.reg(train = train, test=test, y = y, k = k)

out_of_fold_knn<-results_cv$pred
pred_knn<-results$pred

save(out_of_fold_knn,pred_knn,k,file="kNNNearest2048_11292017.RData")
