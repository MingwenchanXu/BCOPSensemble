#--------- loading packages ------------
# set.seed(100)
# library(ggplot2)
library(dplyr)
# library(randomForest)
library(kimisc)
library(glmnet)
library(ranger)
library(MASS)
library(caTools)
library(caret)

#--------- read in simulation data ------------
testData = read.csv("~/Desktop/自学/申请/大三暑假/Research /code/BCOPS/SimTest.csv")[,-1]
trainData = read.csv("~/Desktop/自学/申请/大三暑假/Research /code/BCOPS/SimTrain.csv")[,-1]

#--------- cleaning data ------------
colnames(testData) = c("x1", "x2", "label")
colnames(trainData) = c("x1", "x2", "label")
# getting rid of labels
x.testing = testData[,1:2]
x.train = trainData[,1:2]
colnames(x.train)= colnames(x.testing)
y.train = trainData[,3]
# labels
label.list = sort(unique(y.train))
ytest = testData[,3]


#--------- L-fold for testing data ------------
# BCOPSEnsemble L-fold helper function
# parameters: dat - a data frame
#             num - a positive integer,referring the number of the samples the input data frame will be divided in
# output: index.list, a list of lists of indexes of the objects from input data frame with length(L.Fold(dat, num)) = num.

L.Fold = function(dat, num){
  index.list = list()
  foldid = sample(1:num, nrow(dat), replace = TRUE)
  for (i in 1:num) {
    index.list[[i]] = which(foldid ==i)
  }
  
  return(index.list)
}

#--------- BCOPSEnsemble trainning------------
# BCOPSEnsemble trainging function
# parameters: index.l - a positive integer, referring to which fold you want to train
#             x - training data frame.
#             y - full list of labels from training data.
#             label- list of unique values in y
#             xte - testing data frame (no labels)
#             num.b - a positive integer, number of repetions in training process
# output: model.class.list, a list of lists with length(model.class.list) = length(label). For example,
#         model.class.list[[1]] is built for the first lable in the label list, which is label[1], and
#         has two list components, index and model. The index is a list storing the lists of indexes of samples used
#         in the ranger model with corresponding index. The model is a list of ranger objects.
# example:
# testing.l.index = L.Fold(x.testing, 4)
# index1 = testing.l.index[[1]]
# sample2 = BCOPSEnsemble.trainning(index1, x.train, y.train, label.list, x.testing, 50)

BCOPSEnsemble.trainning = function(index.l, x, y, label, xte, num.b, weightFun = F){
  # determine the how many labels
  K = length(label)
  
  # build the list to store information in different classes
  model.class.list = list()
  
  # filtering out the desired testing data
  xte.l = xte[index.l,]
  # the complement of the desired data
  xte.lc = xte[-index.l,]
  
  
  # loop for training
  for (k in 1:K) {
    # training data with label k
    xk.dat =  x[y==label[k],]
    # size of xK
    row.k = nrow(xk.dat)
    # list for storing ranger models
    model.k = list()
    # list for storing random index used to train the model
    index.k = list()
    
    # loop for training data with a specific class k
    for (b in 1:num.b) {
      # random select samples without replacement
      index.b = sort(sample(1:row.k, 0.6*row.k, replace = F))
      # store the random index
      index.k[[b]]= index.b
      # random sample
      sample.b = xk.dat[index.b, ]
      data.c = x[y!=label[k],]
      index.c = sort(sample(1:nrow(data.c), 0.2*nrow(data.c), replace = F))
      sample.c = x[index.c,]
      
      # combine Ib with the complement of testing.l
      xk = rbind(xte.lc,sample.b)
      yk = c(rep(0, nrow(xte.lc)),rep(1,nrow(sample.b)))
      
      # for weight
      xkw = rbind(xte.lc,sample.b, sample.c)
      ykw = c(rep(0, nrow(xte.lc)),rep(1,nrow(sample.b)), rep(1/K,nrow(sample.c)))

      weightList.log = c(rep(1, nrow(xte.lc)),rep(1,nrow(sample.b)),
                     rep(1 / (log2(nrow(xte.lc)+2)), nrow(sample.c)))
      
      weightList0 = c(rep(1, nrow(xte.lc)),rep(1,nrow(sample.b)),
                     rep(0, nrow(sample.c)))
      weightList1 = c(rep(1, nrow(xte.lc)),rep(1,nrow(sample.b)),
                      rep(1, nrow(sample.c)))
      weightList.sqrt = c(rep(1, nrow(xte.lc)),rep(1,nrow(sample.b)),
                      rep(1/(sqrt(nrow(sample.c))), nrow(sample.c)))
      
      
      if(length(yk) == 0){
        stop(paste0("class ", labels[k], " does not exist in the training data!"))
      }
      
      # tidy the matrix into dataframes
      temp = ncol(xk)
      dat = cbind(xk, yk)
      dat = data.frame(dat)
      colnames(dat) = c(paste0("feature", (1:temp)), "response")
      
      tempw= ncol(xkw)
      datw = cbind(xkw, ykw)
      datw = data.frame(datw)
      colnames(datw) = c(paste0("feature", (1:tempw)), "response")
      
      # store the rager object into the list
      
      if(weightFun == 0){
        model.k[[b]] = ranger(formula = response~., data = datw, case.weights = weightList0,num.trees =  100)
      }else if(weightFun == "log"){
        model.k[[b]] = ranger(formula = response~., data = datw, case.weights = weightList.log,num.trees =  100)
      }else if (weightFun == "sqrt"){
        model.k[[b]] = ranger(formula = response~., data = datw, case.weights = weightList.sqrt,num.trees =  100)
      }else if (weightFun == 1){
        model.k[[b]] = ranger(formula = response~., data = datw, case.weights = weightList1,num.trees =  100)
      }else if (weightFun == F){
        model.k[[b]] = ranger(formula = response~., data = dat, num.trees =  100)
      }
      
      index.model.list = list(index = index.k, model = model.k)
    }
    model.class.list[[k]] = index.model.list
    
  }
  return(model.class.list)
  
}

#--------- extracting models------------
# BCOPSEnsemble Extract.Model helper function
# parameters: lists - output of BCOPSEnsemble.trainning function
# output: model.list, list of lists storing ranger models for each class, length(model.list) = length(label)
#         model.list[[1]] contains the list of ranger models for label1.
# example:
# model.k2 = Extract.Model(sample2)
Extract.Model = function(lists){
  model.list = list()
  for (i in 1:length(lists)){
    model.list[[i]] = lists[[i]]$model
  }
  return(model.list)
}
#--------- extracting index------------
# BCOPSEnsemble Extract.Index helper function
# parameters: lists - output of BCOPSEnsemble.trainning function
# output: index.list, list of lists storing the index of sample used to train ranger models for each class,
#         length(index.list) = length(label)
#         index.list[[1]] contains the list of index of sample used to train ranger models for label1.
# example:
# index.k = Extract.Index(sample2)
Extract.Index = function(lists){
  index.list = list()
  for (i in 1:length(lists)){
    index.list[[i]] = lists[[i]]$index
  }
  return(index.list)
}


#--------- choose index------------
# BCOPSEnsemble BCOPSEnsemble.choose.indexs helper function
# parameters: models - output of BCOPSEnsemble.trainning function
#             x - training data frame.
#             y - full list of labels from training data.
#             label- list of unique values in y
#             num.b - a positive integer, number of repetions in training process
# output: index.list, list of lists storing index of object not being be used to train the bth model for each class,
#         length(index.list) = length(label)
#         the ith list in index.list[[1]] contains the index that ith sample are not used to train the bth model for label1.
# example:
# testing.l.index = L.Fold(x.testing, 4)
# index1 = testing.l.index[[1]]
# sample1 = BCOPSEnsemble.trainning(index1, x.train, y.train, label.list, x.testing, 50)
# index.pred = BCOPSEnsemble.choose.index(sample1, x.train, y.train, label.list,50)

BCOPSEnsemble.choose.index= function(models, x, y, label, num.b){
  # determine the how many labels
  K = length(label)
  
  index.list = list()
  
  # loop for each class
  for (k in 1:K) {
    index.list.subk = list()
    
    # extracting class k data and making them into data frame
    xk = x[y==label[k],]
    
    # extracting index list and model list for class k
    train.index.k = Extract.Index(models)[[k]]
    train.model.k = Extract.Model(models)[[k]]
    
    # determine the ith sample are not used in which process of training process
    for (i in 1:nrow(xk)) {
      filter.list = vector(length = num.b)
      for (b in 1:num.b) {
        index.temp = train.index.k[[b]]
        if(i  %in% index.temp){
          filter.list[b]= T
        }else{
          filter.list[b]= F
        }
      }
      
      index.use = which(filter.list==F)
      
      #storing the desired index in the list
      index.list.subk[[i]] = index.use
    }
    
    index.list[[k]]= index.list.subk
    
  }
  return(index.list)
}



# --------- aggregation vk(except i itself) for trainning ------------
# BCOPSEnsemble.predict.train
# parameters: models - output of BCOPSEnsemble.trainning function
#             index.pred - output of BCOPSEnsemble.choose.index
#             x - training data frame.
#             y - full list of labels from training data.
#             label- list of unique values in y
# output: train.pred.list, list of lists storing the mean prediction score for each object in each class,
#         length(train.pred.list) = length(label)
# example:
# testing.l.index = L.Fold(x.testing, 4)
# index1 = testing.l.index[[1]]
# sample1 = BCOPSEnsemble.trainning(index1, x.train, y.train, label.list, x.testing, 50)
# index.pred = BCOPSEnsemble.choose.index(sample1, x.train, y.train, label.list,50)
# train.pred.matrix = BCOPSEnsemble.predict.train(sample1,index.pred, x.train, y.train, label.list)

BCOPSEnsemble.predict.train= function(models, index.pred, x, y, label){
  K = length(label)
  train.pred.list = list()
  
  for(k in 1:K){
    # extracting class k data and desired part for the remainning procedure
    xk = x[y==label[k],]
    model.full.k = Extract.Model(models)[[k]]
    M = length(model.full.k)
    train.full.matrix = matrix(data = NA, nrow = nrow(xk), ncol = M )
    index.pred.k = index.pred[[k]]
    final.vk = vector(length = nrow(xk))
    
    # tidy the matrix into dataframes
    temp = ncol(xk)
    dat = data.frame(xk)
    colnames(dat) = c(paste0("feature", (1:temp)))
    
    for (m in 1:M) {
      model.m = model.full.k[[m]]
      train.full.matrix[,m] = predict(model.m, dat)$predictions
    }
    
    for(i in 1:nrow(xk)){
      score.full = train.full.matrix[i,]
      index.temp = index.pred.k[[i]]
      score.select = score.full[index.temp]
      final.vk[i] = mean(score.select)
    }
    
    train.pred.list[[k]]= final.vk
  }
  return(train.pred.list)
  
}






# --------- aggregation vk(except i itself) for testing ------------
# BCOPSEnsemble.predict.test
# parameters: train.v - output of BCOPSEnsemble.predict.train
#             models - output of BCOPSEnsemble.trainning function
#             index.pred - output of BCOPSEnsemble.choose.index
#             index.l - a positive integer, referring to which fold you want to train
#             x - training data frame.
#             y - full list of labels from training data.
#             label- list of unique values in y
#             xte - testing data frame (no labels)
#             num.b - a positive integer, number of repetions in training process
# output: xte.comparison.k, a conformal score matrix which the ith row refers to the score for ith label.
# example:
# testing.l.index = L.Fold(x.testing, 4)
# index1 = testing.l.index[[1]]
# sample1 = BCOPSEnsemble.trainning(index1, x.train, y.train, label.list, x.testing, 500)
# index.pred = BCOPSEnsemble.choose.index(sample1, x.train, y.train, label.list,500)
# train.pred.matrix = BCOPSEnsemble.predict.train(sample1,index.pred, x.train, y.train, label.list)
# test.pred.matrix1 = BCOPSEnsemble.predict.test(train.pred.matrix, sample1,index.pred, index1, x.train, y.train, label.list, x.testing, 500)

BCOPSEnsemble.predict.test = function(train.v, models, index.pred, index.l, x, y, label, xte, num.b){
  xte.l = xte[index.l, ]
  K = length(label)
  xte.comparison.k = matrix(data = NA, nrow = nrow(xte.l), ncol = K)
  
  for (k in 1:K) {
    # extracting class k data and making them into data frame
    xk = x[y==label[k],]
    row.k = nrow(xk)
    
    # getting models for k class
    model.full.k = Extract.Model(models)[[k]]
    M = length(model.full.k)
    
    # getting indexs for k class
    index.pred.k = index.pred[[k]]
    # making them to data frame and then matrix for future use
    index.pred.k.matrix = plyr::ldply(index.pred.k, rbind)
    index.pred.k.matrix = data.matrix(index.pred.k.matrix,rownames.force = NA)
    
    # declare full prediction score matrix
    pred.full.matrix = matrix(data = NA, nrow = nrow(xte.l), ncol = M )
    xte.sk = vector(length = nrow(xte.l))
    
    # prediction score for train
    train.vk = train.v[[k]]
    
    # tidy the matrix into dataframes
    temp = ncol(xte.l)
    dat = data.frame(xte.l)
    colnames(dat) = c(paste0("feature", (1:temp)))
    
    # all prediction score
    for (m in 1:M) {
      model.m = model.full.k[[m]]
      pred.full.matrix[,m] = predict(model.m, dat)$predictions
    }
    
    # calling rcpp functions to compute conformal scores
    xte.score = CompareLoop( nrow(xte.l), pred.full.matrix, row.k, index.pred.k.matrix,  train.vk, num.b)
    xte.comparison.k[,k] = xte.score
    
  }
  
  return(xte.comparison.k)
}


#--------- BCOPSEnsemble ------------
# BCOPSEnsemble
# parameters: x - training data frame.
#             y - full list of labels from training data.
#             label- list of unique values in y
#             xte - testing data frame (no labels)
#             num.b - a positive integer, number of repetions in training process
#             L - a positive integer, number of splits
# output: conformal_table, a table for conformal score.
# example:
# final.test1 = BCOPSEnsemble(x.train, y.train, label.list,x.testing, 500,4)
BCOPSEnsemble = function(x, y, label, xte, num.b, L, weightFun = FALSE){
  # L-folds
  testing.l.index = L.Fold(xte, L)
  # k labels
  K = length(label)
  
  conformal.matrix = matrix(data = NA, nrow = nrow(xte), ncol =K)
  
  for (l in 1:L) {
    index.use = testing.l.index[[l]]
    xte.l = xte[index.use, ]
    
    # trainning
    train.models = BCOPSEnsemble.trainning(index.use, x, y, label, xte, num.b, weightFun = weightFun)
    index.to.use = BCOPSEnsemble.choose.index(train.models, x, y, label, num.b)
    train.pred.matrix = BCOPSEnsemble.predict.train(train.models,index.to.use, x, y, label)
    
    # prediction
    test.pred.matrix = BCOPSEnsemble.predict.test(train.pred.matrix, train.models, index.to.use, index.use, x, y, label, xte, num.b)
    conformal.matrix[index.use,] = test.pred.matrix
    
  }
  
  # test.with.conformal = cbind(xte,conformal.matrix)
  return(conformal.matrix)
  
}



#--------- Predicted class ------------
# only for plot use
plot.Pred = function(pre.con, alpha){
  list.l = vector(length = nrow(pre.con))
  xc1 = pre.con[,1]
  xc2 = pre.con[,2]
  for(i in 1:nrow(pre.con)){
    if(xc1[i]>= alpha & xc2[i]>= alpha){
      list.l[i] = "Both"
    }else if(xc1[i]<= alpha & xc2[i]<= alpha){
      list.l[i] = "R"
    }else if(xc1[i]>= alpha){
      list.l[i] = "1"
    }else{
      list.l[i] = "2"
    }
  }
  return(list.l)
}

final.test =  read.csv("/Users/xumingwenchan/Desktop/自学/申请/大三暑假/Research /code/BCOPSEnsemble/Ensemble5W/sufficientTest5/finalTest.csv")[,-1]
final.test0=  read.csv("/Users/xumingwenchan/Desktop/自学/申请/大三暑假/Research /code/BCOPSEnsemble/Ensemble5W/sufficientTest5/finalTest0.csv")[,-1]
final.test.log=  read.csv("/Users/xumingwenchan/Desktop/自学/申请/大三暑假/Research /code/BCOPSEnsemble/Ensemble5W/sufficientTest5/finalTestLog.csv")[,-1]
final.test.sqrt=  read.csv("/Users/xumingwenchan/Desktop/自学/申请/大三暑假/Research /code/BCOPSEnsemble/Ensemble5W/sufficientTest5/finalTestSqrt.csv")[,-1]
final.test1=  read.csv("/Users/xumingwenchan/Desktop/自学/申请/大三暑假/Research /code/BCOPSEnsemble/Ensemble5W/sufficientTest5/finalTest1.csv")[,-1]
# 
# 
# 
# 
# Class = plot.Pred(final.test1,0.05)
# 
# plot.dat = cbind(testData,Class)
# s1 = cbind(testData,Class)
# ggplot(data = plot.dat, mapping = aes(x1, x2, color = Class))+geom_point()
# ggplot(data = plot.dat, mapping = aes(x1, x2, color = label))+geom_point()








