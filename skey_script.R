library(tidyverse)
library(ggplot2)
library(xgboost)
library(lubridate)
library(xts)
library(Matrix)
library(fmsb)
library(caret)
library(kerasR)
library(dplyr)
library(plyr)
library(car)
library(DiagrammeR)



source('C:/Users/sspinetto/Desktop/Machine Learning/xgb_functions.R')

#cleaning up, anayzing
train <- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/train.csv/train.csv", stringsAsFactors = FALSE)
macro <- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/macro.csv/macro.csv", stringsAsFactors = FALSE)
test <- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/test.csv/test.csv", stringsAsFactors = FALSE)
rampjeTest <- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/Russian-Housing-Market-master/submission1.csv", stringsAsFactors = FALSE)
sebiTest <- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/Russian-Housing-Market-master/RussiaSubmission8.csv", stringsAsFactors = FALSE)
sample_submission <- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/Russian-Housing-Market-master/RussiaSubmission4.csv", stringsAsFactors = FALSE)
sebiTast<- read.csv("C:/Users/sspinetto/Desktop/Machine Learning/Russian-Housing-Market-master/RussiaSubmissionWINNING.csv", stringsAsFactors = FALSE)


######################
## determining avg price per sub area and splitting data accordingly into thirds, low class, med class, high class

## just a test -- there will be better metrics but curious to see how accuracy improves or nosedives
 # hello<- as.data.frame(sapply( split(train$price_doc, train$sub_area), mean))
 # hello2<- as.vector(unique(train$sub_area))
 # hello2<- sort(hello2)
 # hello$areas<- hello2
 # hello3 <- hello$`sapply(split(train$price_doc, train$sub_area), mean)`>  140
 # 
 # hello$BIGMONEY <- hello3
 # hello4 <- hello$`sapply(split(train$price_doc, train$sub_area), mean)`< 0.5
 # hello$midclass <-hello4
 # 
 # lowclass <- subset(hello, hello$midclass==TRUE)
 # highclass<- subset(hello, hello$BIGMONEY==TRUE)
 # midclass<- subset(hello, hello$BIGMONEY==FALSE & hello$midclass==FALSE)
 # 
 # bad_areas<- as.list(lowclass$areas)
 # decent_areas<- as.list(midclass$areas)
 # great_areas<- as.list(highclass$areas)
 # 
 # train$sub.area.n < train$sub_area
 # train$sub_area<-car::recode(train$sub_area,'bad_areas="AA"')
 # train$sub_area<-car::recode(train$sub_area,'great_areas="GG"')
 # train$sub_area<-car::recode(train$sub_area,'decent_areas="DD"')
 # 
 # test$sub.area.n < test$sub_area
 # test$sub_area<-car::recode(test$sub_area,'bad_areas="AA"')
 # test$sub_area<-car::recode(test$sub_area,'great_areas="GG"')
 # test$sub_area<-car::recode(test$sub_area,'decent_areas="DD"')

 
 
 

###########################

##no macros vars, returns list with first element containing predictions, second the model used to create em

superBooster <- function(train,test) {
id_test = test$id

y_train <- train$price_doc

x_train <- subset(train, select = -c(id, timestamp, price_doc))
x_test <- subset(test, select = -c(id, timestamp))

len_train <- nrow(x_train)
len_test <- nrow(x_test)

train_test <- rbind(x_train, x_test)

features <- colnames(train_test)

for (f in features) {
  if (class(train_test[[f]])=="factor") {
   # cat("VARIABLE : ",f,"\n")
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.numeric(factor(train_test[[f]], levels=levels))
  }
}

x_train = train_test[1:len_train,]
x_test = train_test[(len_train+1):(len_train+len_test),]

dtrain = xgb.DMatrix(data.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(data.matrix(x_test))

xgb_params = list(
  seed = 3,
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.075,
  objective = 'reg:linear',
  max_depth = 6,
  num_parallel_tree = 1,
  min_child_weight = 1,
  base_score = 7
)
# 
# xgb_params <- list(
#   seed = 0,
#   colsample_bytree = 1:10/10,
#   subsample = 1:10/10,
#   eta = 1:10/15,
#   objective = 'reg:linear',
#   max_depth = 1:25,
#   num_parallel_tree = 1,
#   min_child_weight = 1:10,
#   base_score = 7
# )
# 
res = xgb.cv(xgb_params,             dtrain,
             nrounds=2000,
             nfold=10,
             early_stopping_rounds=20,
             print_every_n = 10,
             verbose= 1,
             maximize=F)

best_nrounds = res$best_iteration

#best_nrounds = 131

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

prediction <- predict(gbdt,dtest)
newList <- list("predictions" = prediction , "boosts" = gbdt)

return(prediction)
}


# version with macro vars

superBoosterMac <- function(train,test,macro) {
  id_test = test$id
 
   macro_preds <- macro[c(
    "timestamp",
    "balance_trade",                       "balance_trade_growth",                "eurrub",                             
    "average_provision_of_build_contract", "micex_rgbi_tr",                       "micex_cbi_tr",                       
    "deposits_rate",                       "mortgage_value",                      "mortgage_rate",                      
    "income_per_cap",                      "rent_price_4.room_bus",               "museum_visitis_per_100_cap",         
    "apartment_build"                    
  )]
   
  x_train <- left_join(train, macro_preds, by = "timestamp") 
  x_test <- left_join(test, macro_preds, by = "timestamp")
  
  y_train <- train$price_doc
  
  x_train <- subset(train, select = -c(id, timestamp, price_doc))
  x_test <- subset(test, select = -c(id, timestamp))
  
  len_train <- nrow(x_train)
  len_test <- nrow(x_test)
  
  train_test <- rbind(x_train, x_test)
  
  features <- colnames(train_test)
  
  for (f in features) {
    if (class(train_test[[f]])=="factor") {
      #cat("VARIABLE : ",f,"\n")
      levels <- unique(train_test[[f]])
      train_test[[f]] <- as.numeric(factor(train_test[[f]], levels=levels))
    }
  }
  
  x_train = train_test[1:len_train,]
  x_test = train_test[(len_train+1):(len_train+len_test),]
  
  dtrain = xgb.DMatrix(data.matrix(x_train), label=y_train)
  dtest = xgb.DMatrix(data.matrix(x_test))
  
  xgb_params = list(
    seed = 4,
    colsample_bytree = 0.7,
    subsample = 0.7,
    eta = 0.075,
    objective = 'reg:linear',
    max_depth = 6,
    num_parallel_tree = 1,
    min_child_weight = 1,
    base_score = 7
  )
  # 
  # xgb_params <- list(
  #   seed = 0,
  #   colsample_bytree = 1:10/10,
  #   subsample = 1:10/10,
  #   eta = 1:10/15,
  #   objective = 'reg:linear',
  #   max_depth = 1:25,
  #   num_parallel_tree = 1,
  #   min_child_weight = 1:10,
  #   base_score = 7
  # )
  # 
  # res = xgb.cv(xgb_params,             dtrain,
  #              nrounds=2000,
  #              nfold=10,
  #              early_stopping_rounds=20,
  #              print_every_n = 10,
  #              verbose= 1,
  #              maximize=F)
  # 
  # best_nrounds = res$best_iteration
  
  best_nrounds = 171
  
  gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
  
  prediction <- predict(gbdt,dtest)
  newList <- list("predictions" = prediction , "boosts" = gbdt)
  
  return(newList)
}

vif_func<-function(in_frame,thresh=10,trace=T,...){
  
  require(fmsb)
  
  if(class(in_frame) != 'data.frame') in_frame<-data.frame(in_frame)
  
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  var_names <- names(in_frame)
  for(val in var_names){
    regressors <- var_names[-which(var_names == val)]
    form <- paste(regressors, collapse = '+')
    form_in <- formula(paste(val, '~', form))
    vif_init<-rbind(vif_init, c(val, VIF(lm(form_in, data = in_frame, ...))))
  }
  vif_max<-max(as.numeric(vif_init[,2]), na.rm = TRUE)
  
  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(var_names)
  }
  else{
    
    in_dat<-in_frame
    
    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      
      vif_vals<-NULL
      var_names <- names(in_dat)
      
      for(val in var_names){
        regressors <- var_names[-which(var_names == val)]
        form <- paste(regressors, collapse = '+')
        form_in <- formula(paste(val, '~', form))
        vif_add<-VIF(lm(form_in, data = in_dat, ...))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2]), na.rm = TRUE))[1]
      
      vif_max<-as.numeric(vif_vals[max_row,2])
      
      if(vif_max<thresh) break
      
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }
      
      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]
      
    }
    
    return(names(in_dat))
    
  }
  
}


# ### non split version, no loops
# myPreds <- superBooster(train,test)
# 
# submission <- data.frame(
#   id = test$id,
#   price_doc = myPreds$predictions
# )
# 
# # parameter_table(myPreds$boosts)
# histogram(submission$price_doc)
# # rmse_plot(myPreds$boosts)
# summary(submission$price_doc)
# summary(sample_submission$price_doc)

# 
#loop version w/splits
#train$checker <- train$full_sq / train$price_doc
#test$checker <- test$full_sq / test$price_doc

 train <- split(train,  is.na(train$num_room))
 test <- split(test, is.na(test$num_room))

 loop_test_data <- vector("list", length(train))

 for(x in seq_along(train)){

   loop_test_data[[x]] <- superBooster(train[[x]], test[[x]])
 }

 
  pricePredictions <- ldply(loop_test_data , data.frame)
  Ids <- ldply(test, data.frame)
  

  submission<- data.frame(
    id = Ids$id,
    price_doc = pricePredictions$X..i..
  )
  
  submission <- submission[order(submission$id),]
  
  histogram(submission$price_doc, main="Sanity check: histogram of 'boost1' predictions")

  
  
  summary(submission$price_doc)
  
  # extra sanity checks
  # sebiTast$newPrice <- submission$price_doc
  # sebiTast$oldPrice <- sample_submission$price_doc
  # sebiTast$difference2 <- (as.numeric(sebiTast$price_doc) - as.numeric(sebiTast$newPrice))
  # sebiTast$differenceOLD <- (as.numeric(sebiTast$oldPrice) - as.numeric(sebiTast$newPrice))
  
  write.csv(submission, "C:/Users/sspinetto/Desktop/Machine Learning/Russian-Housing-Market-master/RussiaSubmission22.csv", row.names=F)
