## Source code of the CalPM project

# loading all libraries and files

# List of required packages !!!! please add your libraries to the vector !!!!
required_packages <- c("tidyverse", "data.table", "dplyr",
                       "ggpubr", "ranger", "modeldata", "tidymodels",
                       "rpart.plot", "readr","vip", "ggthemes", 
                       "parsnip", "GGally", "skimr", "xgboost",
                       "doParallel", "kernlab")  

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

#set working directory to the one where document exists
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#print(getwd())

#don't modify the ops file as this is our input we should always derive from
load("ops.RData") ; ops <-
  ops |>
  na.omit() |> 
  select(-ops_pm10, -pres_sea) #external requirement to never use ops_pm10

#validation data (different measurement method)

load("data_test.RData")

ops_validation <- left_join(ops_data, bam, by = "date")|> 
  select(!grimm_pm10, -(poj_h:hour)) |> 
  relocate(bam_pm10, .before = "n_0044") |> 
  rename(grimm_pm10 = bam_pm10)

# Data preparation -------------------------------------------------------------
# Transforming wind to qualitative variable
wind_set_dir <- function(kat) {
  directions <- c("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
  index <- floor((kat + 11.25) / 22.5) %% 16 + 1
  
  directions[index]
}

ops <- ops |> 
  mutate(wd = wind_set_dir(wd))

# Data Split -------------------------------------------------------------------
set.seed(123)
split <- initial_split(data = ops, prop = 3/4, strata = "grimm_pm10")
train_data <- training(split)
test_data <- testing(split)
val_set <- vfold_cv(data = train_data, v = 10, strata = "grimm_pm10")

train_data |> dim()
test_data |> dim()
val_set |> dim()

# Hellwig method implementation ------------------------------------------------
load(file = "hlwg_data.RData")

# Calculating number of all possible combinations of variables (excluding date and second pm10 measure)
(hlwg_comb_count <- (2^length(setdiff(names(ops %>% select(-c(date))), "grimm_pm10"))) - 1)

# Defining the hellwig() function
hellwig <- function(y, x, method = "pearson") {
  requireNamespace("utils")
  x <- x[sapply(x, is.numeric)]
  x <- as.data.frame(x)
  
  # Calculate correlation matrix
  cm <- stats::cor(x, method = method)
  cd <- stats::cor(x, y, method = method)
  
  k <- sapply(seq(2, ncol(x)), function(i) utils::combn(ncol(x), i, simplify = FALSE))
  k <- do.call("c", k)
  
  hfun <- function(v) {
    sapply(v, function(i) cd[i]^2 / sum(abs(cm[v, i])))
  }
  
  all_combinations <- lapply(k, function(comb) {
    score <- sum(hfun(comb))
    list(combination = paste(names(x)[comb], collapse = "-"), score = score)
  })
  
  all_combinations_df <- data.frame(
    combination = sapply(all_combinations, `[[`, "combination"),
    score = sapply(all_combinations, `[[`, "score"),
    stringsAsFactors = FALSE
  )
  
  return(all_combinations_df)
}

# Defining the target variable and the predictor set
hlwg_target_var <- c("grimm_pm10")
hlwg_predictor_set <- setdiff(names(ops %>% select(-c(date))), hlwg_target_var)
hlwg_predictor_data <- ops[, hlwg_predictor_set]
hlwg_target_data <- ops[[hlwg_target_var]]

# Calculating estimated information capacity factor for each possible model
hlwg_all_models <- hellwig(hlwg_target_data, hlwg_predictor_data, method = "pearson")

# Selection of variables in three best potential models
hlwg_best_variables <- unlist(unique(flatten(strsplit(hlwg_all_models[1:3,]$combination, '-')))) 

# Selection of rejected variables
hlwg_rejected_predictors <- setdiff(names(ops %>% select(-c(date, grimm_pm10))), hlwg_best_variables)

# Best model selection
hlwg_best_model <- hlwg_all_models |> slice(which.max(score))
hlwg_best_model

# hlg_vars <- ls()[grep("hellwig|hlwg", ls())]
# for (var in hlg_vars) {
#   assign(var, get(var))
# }
# save(list = hlg_vars, file = "hlwg_data.RData")

# Predictors investigation (GGally) --------------------------------------------



# Initial recipe ---------------------------------------------------------------

ops_rec <- recipe(grimm_pm10  ~., data = train_data) |> 
  step_time(date, features = c("hour")) |>
  step_rm(date) |> 
  step_dummy(all_nominal_predictors()) |> # wd needs to be numeric 
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors()) # Normalizacja

ops_rec |> prep() |>  bake(train_data) |> glimpse() 

# GLM --------------------------------------------------------------------------
load(file = "glm_data.RData")

# Defining formula based on Hellwig's algorithm result
glm_hellwig_vars <- strsplit(c(hlwg_best_model$combination, "date"), "-")
glm_formula <- paste("grimm_pm10 ~", paste(paste(unlist(glm_hellwig_vars)), collapse = " + "))

# Defining GLM Recipe
glm_rec <-
  recipe(as.formula(glm_formula), data = train_data) |> 
  update_role(grimm_pm10, new_role = "outcome") |>
  step_time(date, features = c("hour")) |>
  step_rm(date) |> 
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |> 
  step_interact(terms = ~ starts_with("n_"):starts_with("n_")) |>  
  step_poly(unlist(glm_hellwig_vars[1]), degree = 2, skip = FALSE) |> 
  step_normalize(all_predictors()) |>
  check_missing()
  
prepped_glm_rec <- prep(glm_rec)
juice(prepped_glm_rec) |> glimpse()

# Defining GLM Model
glm_mod <- 
  # linear_reg(penalty = tune(), mixture = 1) |> # For Lasso Regression
  linear_reg(penalty = tune(), mixture = tune()) |>
  set_engine(engine = "glmnet", num.threads = parallel::detectCores() - 1) |> 
  set_mode("regression")

# Defining GLM Workflow
glm_work <- 
  workflow() |> 
  add_model(glm_mod) |> 
  add_recipe(glm_rec)

# Setting up GLM grid
set.seed(123)
glm_grid <- grid_random(penalty(range = c(-5, 1)), mixture(range = c(0.7, 1)), size = 200)
# glm_grid <- grid_random(penalty(range = c(-5, 1)), size = 200) # For Lasso Regression

# GLM grid tuning
glm_res <- try(
  tune_grid(
    glm_work,
    resamples = val_set,
    grid = glm_grid,
    control = control_grid(save_pred = TRUE, allow_par = TRUE),
    metrics = metric_set(rsq, mae)
  )
)

# Assesment of best hyperparameter sets
glm_best <- select_best(glm_res, metric = "rsq")

# GLM: Show hyperparameter fitting results -------------------------------------
# Show 95th hyperparameter quantile
glm_res |>
  show_best(metric = "rsq", n = Inf) |>
  arrange(desc(mean)) |> 
  filter(mean > quantile(mean, 0.95))

# Best hyperparameter result
glm_res |>
  show_best(metric = "rsq", n = Inf) |>
  filter(.config == glm_best$.config)

# GLM: Final GLM Fitting -------------------------------------------------------
# Final GLM Workflow
glm_final_work <- finalize_workflow(glm_work, glm_best)

# Final GLM Fit
glm_fit <- 
  glm_final_work |> 
  fit(data = train_data)

# Final Model Fit Coefficients
glmnet_model <- extract_fit_parsnip(glm_fit)
coef(glmnet_model$fit, s = min(glmnet_model$fit$lambda))

# GLM: GLM Metric Test ---------------------------------------------------------
# Ex-Post Analysis
glm_test_metrics <- glm_fit |> 
  predict(test_data) |> 
  bind_cols(test_data) |> 
  metrics(truth = grimm_pm10, estimate = .pred)

glm_test_metrics # Highest R-Squared = 0.961 (achieved on data split seed = 123)

# glinm_vars <- ls()[grep("glm", ls())]
# for (var in glinm_vars) {
#   assign(var, get(var))
# }
# save(list = glinm_vars, file = "glm_data.RData")


## Random Forest model ---------------------------------------------------------
# loading heavy to compute data-sets and final metrics 
load("rf_data.RData")

# setting up grid and model

rf_grid <- grid_regular(mtry(c(2,19)), 
                        trees(c(400,1500)),
                        levels = 5)

rf_tune_spec <- 
  rand_forest(
    mtry = tune(), 
    trees = tune(),
    min_n = 2) |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_wf <- 
  workflow() |> 
  add_model(rf_tune_spec) |> 
  add_recipe(ops_rec)

#turning on multicore processing for expensive workload
cl <- makeCluster(parallel::detectCores())
registerDoParallel(cl)

rf_fit <-
  rf_wf |>
  tune_grid(
    resamples = vfold_cv(train_data, v=10, repeats=5),
    grid = rf_grid,
    control = control_grid(verbose = TRUE)
  )

#turning off multicore processing
stopCluster(cl)
registerDoSEQ()


rf_best_params <- rf_fit |> 
  select_best(metric = "rsq")

rf_final_wf <- finalize_workflow(
  rf_wf,
  rf_best_params
)

rf_final_fit <- fit(rf_final_wf, data = train_data)

rf_predictions <- predict(rf_final_fit, new_data = test_data)

rf_results <- test_data |> 
  mutate(
    .pred = rf_predictions$.pred) |> 
  select(date, grimm_pm10, .pred)

rf_final_metrics <- rf_predictions |> 
  bind_cols(test_data) |> 
  metrics(truth = grimm_pm10, estimate = .pred)


ggplot(rf_results, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

save(rf_fit,
     rf_final_fit,
     rf_final_metrics,
     file = "rf_data.RData")


## Support Vector machine model , SVM ----------------------------

#cel: znalezienie hiperpłaszczyzny maksymalnie separującej dane, co minimalizuje błędy predykcji.
#model SVM
load("SVM_data.RData")

SVM_r_mod <- svm_rbf(
  mode= "regression",
  cost = tune(), # koszt, uniknięcie nadmiernegi dopasowania
  rbf_sigma = tune() # parametr jądra radialnego RBF
) |> 
  set_engine("kernlab")

# Workflow
SVM_wf <- 
  workflow() |> 
  add_model(SVM_r_mod) |> 
  add_recipe(ops_rec)

# Tuning grid, SVM
SVM_grid <- grid_regular(
  cost(), # zakres kosztów np: cost(range = c(-5, 5)),
  rbf_sigma(), # Zakres sigma dla jądra RBF np  rbf_sigma(range = c(-5, 5)),
  levels = 5)

# Metrics
SVM_metrics <- 
  yardstick::metric_set(
    rsq,
    mae,
    rmse)

# 10 Cross validation 
set.seed(123)
SVM_folds <- vfold_cv(
  train_data, 
  v = 10, 
  repeats = 5)

# Parallel computing 
cores = detectCores(logical = FALSE) - 1
cl = makeCluster(cores)
registerDoParallel(cl)

# Hyperparameter tuning
SVM_tune <- tune_grid(
  object = SVM_wf,
  resamples = SVM_folds,
  grid = SVM_grid,
  metrics = SVM_metrics,
  control = control_grid(verbose = TRUE)
)

stopCluster(cl)

#Showing the best candidates for model
top_SVM_models <- 
  SVM_tune |> 
  show_best(metric="rsq", n = Inf) |> 
  arrange(cost) |> 
  mutate(mean = mean |> round(x = _, digits = 3))

top_SVM_models |> gt::gt()

#5 the best models
SVM_tune|> 
  show_best(metric="rsq") |> 
  knitr::kable()

#The best params
SVM_best_params <- SVM_tune |> 
  select_best(metric = "rsq")

# Final Model
SVM_model_final <- SVM_r_mod |>  
  finalize_model(SVM_best_params)

SVM_final_wf <- finalize_workflow(
  SVM_wf,
  SVM_best_params
)


#Fitting with train data
SVM_fit <- SVM_final_wf |>
  fit(data = train_data)

# Predictions
SVM_predictions <- predict(SVM_fit, new_data = test_data)

SVM_results <- test_data |> 
  mutate(
    .pred = SVM_predictions$.pred) |> 
  select(date, grimm_pm10, .pred)

# Showing metrics of predictions 
SVM_metrics(SVM_results, truth = grimm_pm10, estimate = .pred)
#rsq=0.971, wskazuje na bardzo dobre odwzorowanie danych

# Plot
ggplot(SVM_results, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

save(SVM_tune,
     SVM_fit,
     file = "SVM_data.RData")

## XGBoost model ---------------------------------------------------------------





# XGBoost model specification
xgboost_model <- 
  boost_tree(
    mode = "regression",
    trees = 200,
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
    levels = 7)


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
  vfold_cv(train_data, v=10)


# Set to FALSE to re-tune the XGBoost model, or TRUE to load a previously tuned model from an RData file.
xgboost_load_data = TRUE

# It is better to load stored data, computation takes too long
if (xgboost_load_data) {
  
  load("xgboost_data.RData")
  
} else {
  
# Parallel computing 
  cores = detectCores(logical = FALSE) - 1
  cl = makeCluster(cores)
  registerDoParallel(cl)

# Hyperparameter tuning
  xgboost_tuned <- 
    tune_grid(
    object = xgboost_wf,
    resamples = xgboost_folds,
    grid = xgboost_grid,
    metrics = xgboost_metrics,
    control = control_grid(verbose = TRUE)
)

  stopCluster(cl)
}


# Selecting best parameters
xgboost_tuned |> 
  show_best(metric = "rsq") |> 
  knitr::kable()

xgboost_best_params <- xgboost_tuned |> 
  select_best(metric = "rsq")


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
xgboost_predictions <- predict(xgboost_final_fit, new_data = test_data)

xgboost_results <- test_data |> 
  mutate(
    .pred = xgboost_predictions$.pred) |> 
  select(date, grimm_pm10, .pred)

# Showing metrics of predictions 
xgboost_metrics(xgboost_results, truth = grimm_pm10, estimate = .pred)

# Plot
ggplot(xgboost_results, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

# Saving data
save(xgboost_tuned,
     file = "xgboost_data.RData")
