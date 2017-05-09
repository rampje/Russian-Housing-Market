# xgboost helper functions

rmse_plot <- function(xgb_model){
  #require(ggplot2)
  
  plot_data <- xgb_model$evaluation_log
  
  plot(plot_data$iter, plot_data$train_rmse, type = "o")
  
  # ggplot implementation
  #ggplot(plot_data, aes(iter, train_rmse)) + geom_point()
}

parameter_table <- function(xgb_model){
  
  params <- xgb_model$params
  params <- unlist(params)
  
  rmse_all <- xgb_model$evaluation_log
  rmse_final <- rmse_all$train_rmse[nrow(rmse_all)]
  
  data.frame(
    "parameter" = c(names(params), 
                    "train rmse",
                    "iterations"),
    
    "value" = c(as.character(params), 
                rmse_final,
                nrow(rmse))
  )
}

# this needs to be a list since parameters don't have equal numbers
# of configurations (tree depth vs eta for example)

# the idea is to incorporate these parameters to a larger function which
# runs xgboost using all configurations.
xgb_parameters <- list(
  #seed = 0,
  #colsample_bytree = 1:10/10,
  subsample = 1:10/10,
  eta = 1:10/15,
  objective = 'reg:linear',
  max_depth = 1:25,
  #num_parallel_tree = 1,
  min_child_weight = 1:10
  #base_score = 7
)

# need to exploit "num_parallel_tree" parameter to build random forests

multiple_xgb <- function(xgb_params){
  
}
