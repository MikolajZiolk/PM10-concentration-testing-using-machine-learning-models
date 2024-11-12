## Source code of the CalPM project

# loading all libraries and files

# List of required packages !!!! please add your libraries to the vector !!!!
required_packages <- c("tidyverse", "data.table", "dplyr",
                       "ggpubr", "ranger", "modeldata", "tidymodels",
                       "rpart.plot", "readr","vip", "ggthemes", 
                       "parsnip", "GGally", "skimr", "xgboost")  

# Function to run packages and install missing ones
install_if_missing <- function(packages) {
  missing_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(missing_packages)) {
    install.packages(missing_packages)
  }
  invisible(lapply(packages, library, character.only = TRUE))
}

install_if_missing(required_packages)
tidymodels_prefer()
set.seed(123)
#set working directiory to the one where document exists
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#don't modify the ops file as this is our input we should always derive from
load("ops.RData") ; ops <- ops |> na.omit()
#ops |> glimpse()



##data preparation
#transforming wind to quality variable
wind_set_dir <- function(kat) {
  directions <- c("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
  index <- floor((kat + 11.25) / 22.5) %% 16 + 1
  
  directions[index]
}

ops <- ops |> 
  mutate(wd = wind_set_dir(wd))

##Hellwig method implementation


#Preparing recipe to train models

split <- initial_split(data = ops, prop = 3/4, strata = "grimm_pm10")
train_data <- training(split)
test_data <- testing(split)



ops_rec <- recipe(grimm_pm10  ~., data = train_data) |> 
  update_role(ops_pm10, new_role = "ID") |> 
  step_time(date, features = c("hour")) |>
  step_rm(date) |> 
  step_dummy(all_nominal_predictors()) |> # wd needs to be numeric 
  step_zv(all_predictors())


ops_rec |> prep() |>  bake(train_data) |> glimpse()



## Linear regression model



## Random Forest model



## Support Vector machine model



## XGBoost model

# XGBoost model specification
xgboost_model <- 
  boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) |> 
  set_engine("xgboost")


# Tuning grid 
xgboost_grid <- 
  grid_regular(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction(),
    levels = 3)


# Workflow
xgboost_wf <- 
  workflow() |> 
  add_model(xgboost_model) |> 
  add_recipe(ops_rec)


# Metrics
xgboost_metrics <- 
  yardstick::metric_set(
    rsq,
    mae,
    rmse)

# Cross validation 
xgboost_folds <- 
  vfold_cv(train_data, v=5)


# Hyperparameter tuning
xgboost_tuned <- 
  tune_grid(
  object = xgboost_wf,
  resamples = xgboost_folds,
  grid = xgboost_grid,
  metrics = xgboost_metrics,
  control = control_grid(verbose = TRUE)
)

# Selecting best parameters
xgboost_tuned |> 
  show_best(metric = "rmse") |> 
  knitr::kable()

xgboost_best_params <- xgboost_tuned |> 
  select_best(metric = "rmse")

# Final Model
xgboost_model_final <- xgboost_model |>  
  finalize_model(xgboost_best_params)


xgboost_final_wf <- finalize_workflow(
  xgboost_wf,
  xgboost_best_params
)

# Fit
xgboost_final_fit <- fit(xgboost_final_wf, data = train_data)

# Predictions
xgboost_predictions <- predict(final_fit, new_data = test_data)

xgboost_results <- test_data |> 
  mutate(
    .pred = xgboost_predictions$.pred) |> 
  select(date, grimm_pm10, .pred)

ggplot(xgboost_results, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

save(xgboost_model,
     xgboost_grid,
     xgboost_wf,
     xgboost_metrics,
     xgboost_folds,
     xgboost_tuned,
     xgboost_best_params,
     xgboost_model_final,
     xgboost_final_wf,
     xgboost_final_fit,
     xgboost_predictions,
     xgboost_results,
     file = "xgboost_data.RData")
