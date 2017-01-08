#########################
#This generates predictions using XGBoost, it supplied both predictions
#on the test data and out of fold predictions for the training data
#input: Raw CSV supplied by Allstate
#output: Test and out of fold train predictions
#
#Code based off of: 
#https://www.kaggle.com/tilii7/allstate-claims-severity/bias-correction-xgboost
#https://www.kaggle.com/misfyre/allstate-claims-severity/encoding-feature-comb-modkzs-1108-72665/run/445260

library(data.table)
library(dplyr)
library(Matrix)
library(xgboost)
library(Metrics)
library(forecast)
library(caret)
library(ggplot2)

##########################################
##Functions
##########################################

##########################################
##Function to correct skewness in data using BoxCox Transform
munge_skewed <- function(train, test,num_feat,skew_threshold){
  test[,loss:=0]
  train_test<-rbind(train,test)
  ##Find which features are above skew threshold
  skewed_feats<- sapply(train_test[,names(train_test) %in% num_feat, with=FALSE], skewness)
  cat("Skew in features:",skewed_feats,"\n")
  skewed_feats = skewed_feats[abs(skewed_feats) > skew_threshold]
  skewed_feats = names(skewed_feats)

  for (name in skewed_feats){
    cat("Unskewing: ", name)
    train_test[[name]]=train_test[[name]]+1
    lambda = BoxCox.lambda(train_test[[name]])
    train_test[[name]]=BoxCox(train_test[[name]],lambda)
  }
  return (train_test)
}

##########################################
##Function to calculate MAE
xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y)-shift,exp(yhat)-shift)
  err = round(err, digits =5)
  return (list(metric = "error", value = err))
}

##########################################
##Custom Objective Function
logregobj <- function (preds, dtrain){
  labels = getinfo(dtrain, "label")
  con = 2
  x = preds-labels
  grad = (con*x) / (abs(x)+con)
  hess = (con^2) / (abs(x)+con)^2
  return (list(grad = grad, hess = hess))
}


##########################################
##Load and Clean Data
##########################################

#########################################
#Files to Open
TRAIN_FILE = "~/Desktop/kaggle/AllState/input/train.csv"
TEST_FILE = "~/Desktop/kaggle/AllState/input/test.csv"
SUBMISSION_FILE = "~/Desktop/kaggle/AllState/input/sample_submission.csv"

##########################################
##Param for model
shift = 200
skew_threshold = 0.25
comb_feat = c('cat80','cat87','cat57','cat12','cat79','cat10','cat7','cat89','cat2','cat72','cat81',
              'cat11','cat1','cat13','cat9','cat3','cat16','cat90','cat23','cat36','cat73','cat103',
              'cat40','cat28','cat111','cat6','cat76','cat50','cat5','cat4','cat14','cat38','cat24','cat82','cat25')


##########################################
#Read in Files
train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

num_feat=names(train)[startsWith(names(train),"cont")]
cat_feat=names(train)[startsWith(names(train),"cat")]

##########################################
##Unskew data
ntrain=dim(train)[1]
train_test = munge_skewed(train,test, num_feat,skew_threshold)

rm(train,test)
gc()

##########################################
#Scale numeric values (mean of 0)
for (feat in num_feat){
  train_test[[feat]]<-scale(train_test[[feat]])
}

##########################################
#Create all possible combinations of categorical features to be added together
comb <- t(combn(comb_feat,2))

for (i in 1:dim(comb)[1]){
  feat <-paste0(comb[i,1], "_", comb[i,2]) #Name of new feature
  new_col<-paste0(train_test[[comb[i,1]]] , train_test[[comb[i,2]]])#new col
  train_test[[feat]]<- paste0(train_test[[comb[i,1]]] , train_test[[comb[i,2]]])
  #encode data to integer for trees
  train_test[[feat]]<-as.integer(factor(train_test[[feat]]))
}

##########################################
#Convert remaining categorical values to integers for model
for (feat in cat_feat){
  train_test[[feat]]<-as.integer(factor(train_test[[feat]]))
}

##########################################
#Apply shift to loss values
train_test[['loss']]=log(train_test[['loss']]+shift)

##########################################
#Seperate data
train<-train_test[1:ntrain,]
test<-train_test[(ntrain+1):dim(train_test)[1],]
test[,loss:=NULL]
rm(train_test)
gc()

cat('Median Loss:', median(train$loss))
cat('Mean Loss: ', mean(train$loss))
save(train,test,shift,comb_feat,skew_threshold, file="11152016-ProcessedTrainTest.Rdata")


##########################################
#Modelling
##########################################

##########################################
#Parameters for XGB
set.seed(1066)
xgb_params = list(
  colsample_bytree= 0.7,
  subsample = 0.7,
  eta = 0.03,
  #objective= 'reg:linear',
  max_depth= 12,
  min_child_weight= 100,
  booster= 'gbtree'
)

nrounds<-10000
kfolds<-2

##########################################
#Perform K fold predictions

#create folds
folds<-createFolds(train$loss, k = kfolds, list = TRUE, returnTrain = FALSE)

#convert test to xgbmatrix
dtest = xgb.DMatrix(as.matrix(test))
i <- 1

#data frame to store predictions
allpredictions <- NULL
prediction_names <- NULL
out_of_fold <- data.frame(matrix(ncol = 1, nrow = nrow(train)))
train_cv<- data.frame(matrix(ncol = 3, nrow = 0))
#feat_imp<-list()
#feat_names<-names(test)
names(out_of_fold)<-c("Prediction")

for (fold in folds){
  cat("Performing fold: ", i)

  x_train<-train[!fold,] #Train set
  x_val<-train[fold,] #Out of fold validation set

  y_train<-x_train$loss
  x_train[,loss:=NULL]

  y_val<-x_val$loss
  x_val[,loss:=NULL]

  #convert to xgbmatrix
  dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
  dval = xgb.DMatrix(as.matrix(x_val), label=y_val)
  feat_names<-names(x_train)
  rm(x_train,y_train,x_val,y_val)
  gc()

  #perform training
  gbdt = xgb.train(params = xgb_params,
               data = dtrain,
               obj=logregobj,
               nrounds = nrounds,
               watchlist = list(train = dtrain, val=dval),
               early_stopping_rounds=50,
               print_every_n = 10,
               feval=xg_eval_mae,
               maximize=FALSE)


  #perform prediction
  allpredictions<-as.data.frame(cbind(allpredictions,predict(gbdt,dtest)))
  prediction_names <-c(prediction_names ,paste0("Prediction_",i))
  
  out_of_fold[fold,] <- exp(predict(gbdt,dval))-shift
  #out_of_fold[fold,] <- (predict(gbdt,dval))
  train_cv<-rbind(train_cv,c(gbdt$best_iteration, gbdt$best_score, gbdt$best_ntreelimit ))
  
  #feat_imp<-list(feat_imp,xgb.importance(feature_names=feat_names,model=gbdt))
  imp<-xgb.importance(feature_names = feat_names, model = gbdt)
  rm(dtrain, dval, gbdt)
  gc()
  i=i+1
}
names(allpredictions)=prediction_names
names(train_cv)<-c("best_iteration","best_score","best_ntreelimit")

##########################################
#Write Results to file
##########################################

submission_mean = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission_median <- submission_mean

submission_mean$loss = exp(rowMeans(allpredictions))-shift
submission_median$loss = exp(apply(allpredictions,1,median))-shift

write.csv(submission_mean,'sub_mean.csv',row.names = FALSE)
write.csv(submission_median,'sub_median.csv',row.names = FALSE)
save(out_of_fold,xgb_params,kfolds,train_cv,allpredictions,folds,shift,file="XGBoost_OutOfFold.RData")

#plot to look at feature importance
xgb.ggplot.importance(imp, top_n=50)
