require(xgboost)
require(Matrix)


train <- read.csv("train.csv")
test <- read.csv("test.csv")
macro <- read.csv("macro.csv")
sample_submission <- read.csv("sample_submission.csv")

id_test <- test$id

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

x_train <- train_test[1:len_train,]
x_test <- train_test[(len_train+1):(len_train+len_test),]

dtrain <- xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest <- xgb.DMatrix(as.matrix(x_test))


xgb_params <- list(
  seed = 0,
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.075,
  objective = 'reg:linear',
  max_depth = 6,
  num_parallel_tree = 1,
  min_child_weight = 1,
  base_score = 7
)

#res = xgb.cv(xgb_params,
#             dtrain,
#             nrounds=2000,
#             nfold=10,
#             early_stopping_rounds=20,
#             print_every_n = 10,
#             verbose= 1,
#             maximize=F)

#best_nrounds = res$best_iteration

best_nrounds <- 10

gbdt <- xgboost(params = xgb_params, 
                 data = dtrain, 
                 nrounds = best_nrounds)

prediction <- predict(gbdt,dtest)
sample_submission$price_doc <- prediction

write.csv(sample_submission, "submission2.csv", row.names = FALSE)
