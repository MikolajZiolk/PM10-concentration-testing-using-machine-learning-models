## Source code of the CalPM project

# loading all libraries and files

# List of required packages !!!! please add your libraries to the vector !!!!
required_packages <- c("tidyverse", "data.table", "dplyr",
                       "ggpubr", "ranger", "modeldata", "tidymodels",
                       "rpart.plot", "readr","vip", "ggthemes", 
                       "parsnip", "GGally", "skimr", "xgboost",
                       "doParallel", "kernlab", "ggplot2")  

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

# Data preparation -------------------------------------------------------------
# Transforming wind to qualitative variable
wind_set_dir <- function(kat) {
  directions <- c("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
  index <- floor((kat + 11.25) / 22.5) %% 16 + 1
  
  directions[index]
}

#don't modify the ops file as this is our input we should always derive from
load("ops.RData") ; ops <-
  ops |>
  na.omit() |> 
  select(-ops_pm10, -pres_sea) |> #external requirement to never use ops_pm10
  mutate(wd = wind_set_dir(wd))
  
#validation data (different measurement method)

load("data_test.RData")

ops_validation <- left_join(ops_data, bam, by = "date")|> 
  select(!grimm_pm10, -(poj_h:hour)) |> 
  relocate(bam_pm10, .before = "n_0044") |> 
  rename(grimm_pm10 = bam_pm10) |> 
  na.omit() |> 
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


##Data correlation analysis ---------------------------


predictor_vars <- setdiff(names(ops), c("grimm_pm10", "date", "wd"))

correlation_results <- ops |> 
  select(all_of(predictor_vars), grimm_pm10) |> 
  cor(use = "complete.obs") |> 
  as.data.frame() |> 
  rownames_to_column(var = "variable") |> 
  filter(variable != "grimm_pm10") |> 
  mutate(correlation = abs(grimm_pm10)) |> 
  arrange(desc(correlation))



## Recipes ---------------------------------------------------------------

#basic recipe
ops_rec <- recipe(grimm_pm10  ~., data = train_data) |> 
  step_time(date, features = c("hour")) |>
  step_rm(date) |> 
  step_dummy(all_nominal_predictors()) |> # wd needs to be numeric 
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors()) # Normalizacja

ops_rec |> prep() |>  bake(train_data) |> glimpse() 


#recipe with predictors chosen by hellwig

hellwig_vars_rec <- strsplit(c(hlwg_best_model$combination, "date"), "-")
final_hellwig_formula <- paste("grimm_pm10 ~", paste(paste(unlist(hellwig_vars_rec)), collapse = " + "))

hellwig_rec <-
  recipe(as.formula(final_hellwig_formula), data = train_data) |> 
  update_role(grimm_pm10, new_role = "outcome") |> 
  step_time(date, features = c("hour")) |> 
  step_rm(date) |> 
  step_dummy(all_nominal_predictors()) |> # Encode categorical variables
  step_zv(all_predictors()) |>            # Remove zero-variance predictors
  check_missing()                         # Ensure no missing values

hellwig_rec |> prep() |>  bake(train_data) |> glimpse() 

#recipe made by analyzing data

ops_rec_upgraded <- recipe(grimm_pm10  ~., data = train_data) |> 
  #step_time(date, features = c("hour")) |>
  step_mutate(n_1000 = n_0750 + n_1000) |> 
  step_mutate(n_0200 = n_0100 + n_0120 + n_0140 + n_0200) |> 
  step_rm(prec, mws, n_0750, n_0100, n_0120, n_0140) |>
  update_role(date, new_role = "ID") |> 
  step_dummy(all_nominal_predictors()) |> 
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors())

ops_rec_upgraded |> prep() |>  bake(train_data) |> glimpse() 


### **************** ML MODELS *******************
 # GLM --------------------------------------------------------------------------

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


glm_load_data = TRUE

if (glm_load_data) {
  
  load(file = "glm_data.RData")
  
} else {
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

}

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
  predict(ops_validation) |> 
  bind_cols(ops_validation) |> 
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

set.seed(123)
rf_tune_spec <- 
  rand_forest(
    mtry = tune(), 
    trees = tune(),
    min_n = 2) |> 
  set_engine("ranger", importance = "permutation") |> 
  set_mode("regression")

rf_wf_basic <- 
  workflow() |> 
  add_model(rf_tune_spec) |> 
  add_recipe(ops_rec)

rf_wf_hlwg <- 
  workflow() |> 
  add_model(rf_tune_spec) |> 
  add_recipe(hellwig_rec)

rf_wf_upgraded <- 
  workflow() |> 
  add_model(rf_tune_spec) |> 
  add_recipe(ops_rec_upgraded)

rf_metrics <- 
  yardstick::metric_set(
    rsq, 
    mae,
    rmse)

# Set to FALSE to re-tune the rf model, or TRUE to load a previously tuned model from an RData file.
rf_load_data = TRUE

if (rf_load_data) {
  
  load("rf_data.RData")
  
} else {
  
#turning on multicore processing for expensive workload
  cl <- makeCluster(parallel::detectCores())
  registerDoParallel(cl)
  
  rf_fit_basic <-
    rf_wf_basic |>
    tune_grid(
      resamples = vfold_cv(train_data, v=5, repeats=3),
      grid = rf_grid,
      control = control_grid(verbose = TRUE),
      metrics = rf_metrics
    )
  
  rf_fit_hlwg <-
    rf_wf_hlwg |>
    tune_grid(
      resamples = vfold_cv(train_data, v=5, repeats=3),
      grid = rf_grid,
      control = control_grid(verbose = TRUE),
      metrics = rf_metrics
    )
  
  rf_fit_upgraded <-
    rf_wf_upgraded |>
    tune_grid(
      resamples = vfold_cv(train_data, v=5, repeats=3),
      grid = rf_grid,
      control = control_grid(verbose = TRUE), 
      metrics = rf_metrics
    )
  #turning off multicore processing
  stopCluster(cl)
  registerDoSEQ()
}

rf_best_params_basic <- rf_fit_basic |> 
  select_best(metric = "rsq")

rf_best_params_hlwg <- rf_fit_hlwg |> 
  select_best(metric = "rsq")

rf_best_params_upgraded <- rf_fit_upgraded |> 
  select_best(metric = "rsq")


rf_final_wf_basic <- finalize_workflow(
  rf_wf_basic,
  rf_best_params_basic
)

rf_final_wf_hlwg <- finalize_workflow(
  rf_wf_hlwg,
  rf_best_params_hlwg
)

rf_final_wf_upgraded <- finalize_workflow(
  rf_wf_upgraded,
  rf_best_params_upgraded
)


rf_final_fit_basic <- fit(rf_final_wf_basic, data = train_data)

rf_final_fit_hlwg <- fit(rf_final_wf_hlwg, data = train_data)

rf_final_fit_upgraded <- fit(rf_final_wf_upgraded, data = train_data)


rf_predictions_basic <- predict(rf_final_fit_basic, new_data = ops_validation)

rf_predictions_hlwg <- predict(rf_final_fit_hlwg, new_data = ops_validation)

rf_predictions_upgraded <- predict(rf_final_fit_upgraded, new_data = ops_validation)



rf_results_basic <- ops_validation |> 
  mutate(
    .pred = rf_predictions_basic$.pred) |> 
  select(date, grimm_pm10, .pred)

rf_results_hlwg <- ops_validation |> 
  mutate(
    .pred = rf_predictions_hlwg$.pred) |> 
  select(date, grimm_pm10, .pred)

rf_results_upgraded <- ops_validation |> 
  mutate(
    .pred = rf_predictions_upgraded$.pred) |> 
  select(date, grimm_pm10, .pred)



rf_metrics(rf_results_basic, truth = grimm_pm10, estimate = .pred)

rf_metrics(rf_results_hlwg, truth = grimm_pm10, estimate = .pred)

rf_metrics(rf_results_upgraded, truth = grimm_pm10, estimate = .pred)


# rf_final_metrics_basic <- rf_predictions |> 
#   bind_cols(test_data) |> 
#   metrics(truth = grimm_pm10, estimate = .pred)
# rf_final_metrics
# 
# rf_final_metrics_ <- rf_predictions |> 
#   bind_cols(test_data) |> 
#   metrics(truth = grimm_pm10, estimate = .pred)
# rf_final_metrics
# 
# rf_final_metrics <- rf_predictions |> 
#   bind_cols(test_data) |> 
#   metrics(truth = grimm_pm10, estimate = .pred)
# rf_final_metrics

ggplot(rf_results_upgraded, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

save(rf_fit_basic,
     rf_fit_hlwg,
     rf_fit_upgraded,
     file = "rf_data.RData")


## Support Vector machine model , SVM ----------------------------

#cel: znalezienie hiperpłaszczyzny maksymalnie separującej dane, co minimalizuje błędy predykcji.
#model SVM

SVM_r_mod <- svm_rbf(
  mode= "regression",
  cost = tune(), # koszt, uniknięcie nadmiernegi dopasowania
  rbf_sigma = tune() # parametr jądra radialnego RBF
) |> 
  set_engine("kernlab")

# Workflow
SVM_wf_basic <- 
  workflow() |> 
  add_model(SVM_r_mod) |> 
  add_recipe(ops_rec)

SVM_wf_hlwg <- 
  workflow() |> 
  add_model(SVM_r_mod) |> 
  add_recipe(hellwig_rec)

SVM_wf_upgraded <- 
  workflow() |> 
  add_model(SVM_r_mod) |> 
  add_recipe(ops_rec_upgraded)

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

# Set to FALSE to re-tune the svm model, or TRUE to load a previously tuned model from an RData file.
SVM_load_data = TRUE

if (SVM_load_data) {
  
  load("SVM_data.RData")
  
} else {
  
# Parallel computing 
  cores = detectCores(logical = FALSE) - 1
  cl = makeCluster(cores)
  registerDoParallel(cl)
  
  # Hyperparameter tuning
  SVM_tune_basic <- tune_grid(
    object = SVM_wf_basic,
    resamples = SVM_folds,
    grid = SVM_grid,
    metrics = SVM_metrics,
    control = control_grid(verbose = TRUE)
  )
  
  SVM_tune_hlwg <- tune_grid(
    object = SVM_wf_hlwg,
    resamples = SVM_folds,
    grid = SVM_grid,
    metrics = SVM_metrics,
    control = control_grid(verbose = TRUE)
  )
  
  SVM_tune_upgraded <- tune_grid(
    object = SVM_wf_upgraded,
    resamples = SVM_folds,
    grid = SVM_grid,
    metrics = SVM_metrics,
    control = control_grid(verbose = TRUE)
  )
  stopCluster(cl)

}
#Showing the best candidates for model
top_SVM_models_basic <- 
  SVM_tune_basic |> 
  show_best(metric="rsq", n = Inf) |> 
  arrange(cost) |> 
  mutate(mean = mean |> round(x = _, digits = 3))

top_SVM_models_hlwg <- 
  SVM_tune_hlwg |> 
  show_best(metric="rsq", n = Inf) |> 
  arrange(cost) |> 
  mutate(mean = mean |> round(x = _, digits = 3))

top_SVM_models_upgraded <- 
  SVM_tune_upgraded |> 
  show_best(metric="rsq", n = Inf) |> 
  arrange(cost) |> 
  mutate(mean = mean |> round(x = _, digits = 3))

top_SVM_models_basic |> gt::gt()
top_SVM_models_hlwg |> gt::gt()
top_SVM_models_upgraded |> gt::gt()


#5 the best models
SVM_tune_basic |> 
  show_best(metric="rsq") |> 
  knitr::kable()

SVM_tune_hlwg |> 
  show_best(metric="rsq") |> 
  knitr::kable()

SVM_tune_upgraded |> 
  show_best(metric="rsq") |> 
  knitr::kable()

#The best params
SVM_best_params_basic <- SVM_tune_basic |> 
  select_best(metric = "rsq")

SVM_best_params_hlwg <- SVM_tune_hlwg |> 
  select_best(metric = "rsq")

SVM_best_params_upgraded <- SVM_tune_upgraded |> 
  select_best(metric = "rsq")

# Final Model
SVM_model_final_basic <- SVM_r_mod |>  
  finalize_model(SVM_best_params_basic)

SVM_model_final_hlwg <- SVM_r_mod |>  
  finalize_model(SVM_best_params_hlwg)

SVM_model_final_upgraded <- SVM_r_mod |>  
  finalize_model(SVM_best_params_upgraded)


SVM_final_wf_basic <- finalize_workflow(
  SVM_wf_basic,
  SVM_best_params_basic
)

SVM_final_wf_hlwg <- finalize_workflow(
  SVM_wf_hlwg,
  SVM_best_params_hlwg
)

SVM_final_wf_upgraded <- finalize_workflow(
  SVM_wf_upgraded,
  SVM_best_params_upgraded
)

#Fitting with train data
SVM_fit_basic <- SVM_final_wf_basic |>
  fit(data = train_data)

SVM_fit_hlwg <- SVM_final_wf_hlwg |>
  fit(data = train_data)

SVM_fit_upgraded <- SVM_final_wf_upgraded |>
  fit(data = train_data)

# Predictions
SVM_predictions_basic <- predict(SVM_fit_basic, new_data = ops_validation)

SVM_predictions_hlwg <- predict(SVM_fit_hlwg, new_data = ops_validation)

SVM_predictions_upgraded <- predict(SVM_fit_upgraded, new_data = ops_validation)


SVM_results_basic <- ops_validation |> 
  mutate(
    .pred = SVM_predictions_basic$.pred) |> 
  select(date, grimm_pm10, .pred)

SVM_results_hlwg <- ops_validation |> 
  mutate(
    .pred = SVM_predictions_hlwg$.pred) |> 
  select(date, grimm_pm10, .pred)

SVM_results_upgraded <- ops_validation |> 
  mutate(
    .pred = SVM_predictions_upgraded$.pred) |> 
  select(date, grimm_pm10, .pred)

# Showing metrics of predictions 
SVM_metrics(SVM_results_basic, truth = grimm_pm10, estimate = .pred)

SVM_metrics(SVM_results_hlwg, truth = grimm_pm10, estimate = .pred)

SVM_metrics(SVM_results_upgraded, truth = grimm_pm10, estimate = .pred)

# Plot
ggplot(SVM_results_upgraded, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

save(SVM_tune_basic,
     SVM_tune_hlwg,
     SVM_tune_upgraded,
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
xgboost_wf_basic <- 
  workflow() |> 
  add_model(xgboost_model) |> 
  add_recipe(ops_rec)

xgboost_wf_hlwg <- 
  workflow() |> 
  add_model(xgboost_model) |> 
  add_recipe(hellwig_rec)

xgboost_wf_upgraded <- 
  workflow() |> 
  add_model(xgboost_model) |> 
  add_recipe(ops_rec_upgraded)

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
  cores = detectCores()
  cl = makeCluster(cores)
  registerDoParallel(cl)

# Hyperparameter tuning
  xgboost_tuned_basic <- 
    tune_grid(
    object = xgboost_wf_basic,
    resamples = xgboost_folds,
    grid = xgboost_grid,
    metrics = xgboost_metrics,
    control = control_grid(verbose = TRUE)
)
  
  xgboost_tuned_hlwg <- 
    tune_grid(
      object = xgboost_wf_hlwg,
      resamples = xgboost_folds,
      grid = xgboost_grid,
      metrics = xgboost_metrics,
      control = control_grid(verbose = TRUE)
    )
  
  xgboost_tuned_upgraded <- 
    tune_grid(
      object = xgboost_wf_upgraded,
      resamples = xgboost_folds,
      grid = xgboost_grid,
      metrics = xgboost_metrics,
      control = control_grid(verbose = TRUE)
    )
  stopCluster(cl)
}


# Selecting best parameters
xgboost_tuned_basic |> 
  show_best(metric = "rsq") |> 
  knitr::kable()

xgboost_tuned_hlwg |> 
  show_best(metric = "rsq") |> 
  knitr::kable()

xgboost_tuned_upgraded |> 
  show_best(metric = "rsq") |> 
  knitr::kable()


xgboost_basic_best_params <- xgboost_tuned_basic |> 
  select_best(metric = "rsq")

xgboost_hlwg_best_params <- xgboost_tuned_hlwg |> 
  select_best(metric = "rsq")

xgboost_upgraded_best_params <- xgboost_tuned_upgraded |> 
  select_best(metric = "rsq")

# Final Model
xgboost_model_final_basic <- xgboost_model |>  
  finalize_model(xgboost_basic_best_params)

xgboost_model_final_hlwg <- xgboost_model |>  
  finalize_model(xgboost_hlwg_best_params)

xgboost_model_final_upgraded <- xgboost_model |>  
  finalize_model(xgboost_upgraded_best_params)

xgboost_final_wf_basic <- finalize_workflow(
  xgboost_wf_basic,
  xgboost_basic_best_params
)

xgboost_final_wf_hlwg <- finalize_workflow(
  xgboost_wf_hlwg,
  xgboost_hlwg_best_params
)

xgboost_final_wf_upgraded <- finalize_workflow(
  xgboost_wf_upgraded,
  xgboost_upgraded_best_params
)

# Fit
xgboost_final_fit_basic <- fit(xgboost_final_wf_basic, data = train_data)

xgboost_final_fit_hlwg <- fit(xgboost_final_wf_hlwg, data = train_data)

xgboost_final_fit_upgraded <- fit(xgboost_final_wf_upgraded, data = train_data)

# Predictions
xgboost_predictions_basic <- predict(xgboost_final_fit_basic, new_data = ops_validation)

xgboost_predictions_hlwg  <- predict(xgboost_final_fit_hlwg, new_data = ops_validation)

xgboost_predictions_upgraded  <- predict(xgboost_final_fit_upgraded, new_data = ops_validation)

xgboost_results_basic <- ops_validation |> 
  mutate(
    .pred = xgboost_predictions_basic$.pred) |> 
  select(date, grimm_pm10, .pred)

xgboost_results_hlwg <- ops_validation |> 
  mutate(
    .pred = xgboost_predictions_hlwg$.pred) |> 
  select(date, grimm_pm10, .pred)

xgboost_results_upgraded <- ops_validation |> 
  mutate(
    .pred = xgboost_predictions_upgraded$.pred) |> 
  select(date, grimm_pm10, .pred)

# Showing metrics of predictions 
xgboost_metrics(xgboost_results_basic, truth = grimm_pm10, estimate = .pred)

xgboost_metrics(xgboost_results_hlwg, truth = grimm_pm10, estimate = .pred)

xgboost_metrics(xgboost_results_upgraded, truth = grimm_pm10, estimate = .pred)

# Plot
ggplot(xgboost_results_upgraded, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

# Saving data
save(xgboost_tuned_basic,
     xgboost_tuned_upgraded,
     xgboost_tuned_hlwg,
     file = "xgboost_data.RData")



## Metric table all models  ----------------------------------------------------

# xgboost
xgboost_metrics(xgboost_results_basic, truth = grimm_pm10, estimate = .pred)

xgboost_metrics(xgboost_results_hlwg, truth = grimm_pm10, estimate = .pred)

xgboost_metrics(xgboost_results_upgraded, truth = grimm_pm10, estimate = .pred)


xgboost_metrics_basic <- xgboost_metrics(xgboost_results_basic, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "Basic")
xgboost_metrics_hlwg <- xgboost_metrics(xgboost_results_hlwg, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "HLWG")
xgboost_metrics_upgraded <- xgboost_metrics(xgboost_results_upgraded, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "Upgraded")

# Make one table from all metrics
xgboost_all_metrics <- bind_rows(xgboost_metrics_basic, xgboost_metrics_hlwg, xgboost_metrics_upgraded)


# Table for better comparsion
xgboost_comparison_table <- xgboost_all_metrics |> 
  select(recipe, .metric, .estimate) |> 
  pivot_wider(names_from = .metric, values_from = .estimate)

# SVM
SVM_metrics(SVM_results_basic, truth = grimm_pm10, estimate = .pred)

SVM_metrics(SVM_results_hlwg, truth = grimm_pm10, estimate = .pred)

SVM_metrics(SVM_results_upgraded, truth = grimm_pm10, estimate = .pred)


SVM_metrics_basic <- SVM_metrics(SVM_results_basic, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "Basic")
SVM_metrics_hlwg <- SVM_metrics(SVM_results_hlwg, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "HLWG")
SVM_metrics_upgraded <- SVM_metrics(SVM_results_upgraded, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "Upgraded")

# Make one table from all metrics
SVM_all_metrics <- bind_rows(SVM_metrics_basic, SVM_metrics_hlwg, SVM_metrics_upgraded)


# Table for better comparsion
SVM_comparison_table <- SVM_all_metrics |> 
  select(recipe, .metric, .estimate) |> 
  pivot_wider(names_from = .metric, values_from = .estimate)



# rf
rf_metrics(rf_results_basic, truth = grimm_pm10, estimate = .pred)

rf_metrics(rf_results_hlwg, truth = grimm_pm10, estimate = .pred)

rf_metrics(rf_results_upgraded, truth = grimm_pm10, estimate = .pred)

rf_metrics_basic <- rf_metrics(rf_results_basic, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "Basic")
rf_metrics_hlwg <- rf_metrics(rf_results_hlwg, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "HLWG")
rf_metrics_upgraded <- rf_metrics(rf_results_upgraded, truth = grimm_pm10, estimate = .pred) |> 
  mutate(recipe = "Upgraded")

# Make one table from all metrics
rf_all_metrics <- bind_rows(rf_metrics_basic, rf_metrics_hlwg, rf_metrics_upgraded)


# Table for better comparsion
rf_comparison_table <- rf_all_metrics |> 
  select(recipe, .metric, .estimate) |> 
  pivot_wider(names_from = .metric, values_from = .estimate)


# Final tabels

## Notes ----------------------------------------------------------
# All model metrics are calculated using the ops_validation dataset.
# The file was modified to disable parameter retuning, so the entire script can be executed without adjustments.

# Below are the results for each model.
rf_comparison_table <- rf_comparison_table |> mutate(model = "Random forest")
SVM_comparison_table <- SVM_comparison_table |> mutate(model = "SVM")
xgboost_comparison_table <- xgboost_comparison_table |> mutate(model = "XGBoost")

# Glm used 1 recipe
glm_comparasion_tabel <- glm_test_metrics  |> 
  select(.metric, .estimate) |> 
  pivot_wider(names_from = .metric, values_from = .estimate) |> 
  mutate(model = "GLM", recipe = "GLM recipe")


rf_best_metric <- rf_comparison_table |> 
  slice_max(rsq, n = 1, with_ties = FALSE)

SVM_best_metric <- SVM_comparison_table |> 
  slice_max(rsq, n = 1, with_ties = FALSE)

xgboost_best_metric <- xgboost_comparison_table |> 
  slice_max(rsq, n = 1, with_ties = FALSE)

final_metrics <- bind_rows(glm_comparasion_tabel, rf_best_metric, SVM_best_metric, xgboost_best_metric)


## Comparison Charts  ----------------------------------------------------

#The best versions of models and uniform structure
glm_resultsX <- ops_validation |> 
  mutate(
    model = "GLM", 
    true = grimm_pm10, 
    pred = predict(glm_fit, new_data = ops_validation)$.pred
  ) |> 
  select(date, grimm_pm10, pred, model, true, pred) |> 
  rename(.pred = pred) |> 
  mutate(pred = .pred)

rf_resultsX <- rf_results_upgraded |> 
  mutate(model = "Random Forest", true = grimm_pm10, pred = .pred)

svm_resultsX <- SVM_results_hlwg |> 
  mutate(model = "SVM", true = grimm_pm10, pred = .pred)

xgb_resultsX <- xgboost_results_upgraded |> 
  mutate(model = "XGBoost", true = grimm_pm10, pred = .pred)

# Combination of all results
all_results <- bind_rows(glm_resultsX, rf_resultsX, svm_resultsX, xgb_resultsX)


# Prepare data to tidy
tidy_results <- all_results |> 
  pivot_longer(cols = c(grimm_pm10, .pred), 
               names_to = "type", 
               values_to = "value") |> 
  mutate(type = if_else(type == "grimm_pm10", "True", model))

ggplot(tidy_results, aes(x = date, y = value, color = type)) +
  geom_line() +
  labs(
    title = "Comparison of Actual and Predicted PM10 Values",
    x = "Date",
    y = "PM10 Concentration [µg/m³]",
    color = "Type/Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Point Chart
ggplot(all_results, aes(x = date)) +
  geom_point(aes(y = true, color = "Actual"), size = 1) +
  geom_point(aes(y = pred, color = "Predicted"), shape = 4) +
  facet_wrap(~ model, scales = "free_y") +
  labs(
    title = "Comparison of Actual and Predicted Values",
    x = "Date",
    y = "PM10 Concentration",
    color = "Values"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "yellow", "Predicted" = "red"))
