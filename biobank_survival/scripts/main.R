data_path <- Sys.getenv("BIOBANK_DATA_PATH")
file_path <- file.path(data_path, "first_visit_survival_data_pyppg_embeddings.parquet")
required_packages <- c("arrow", "randomForestSRC", "survival", "ggplot2", "pec", "caret")
for(pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  }
}

cat("Loading data from:", file_path, "...\n")
# Read the Parquet file using the arrow package
df <- arrow::read_parquet(file_path)

# Subset the data: select only observations where MACE_days > 90
df <- subset(df, MACE_days > 90)
cat("Data dimensions after subsetting:", dim(df), "\n")

# Perform a stratified train/test split using the caret package
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(df$MACE, p = 0.7, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]
cat("Training set dimensions:", dim(train), "\n")
cat("Test set dimensions:", dim(test), "\n")

# Get all column names that start with "pca_"
pca_features <- grep("^pca_", names(df), value = TRUE)

# Check the selected features
print(pca_features)

# Create the survival formula using the dynamically generated PCA features
surv_formula <- as.formula(
  paste("Surv(MACE_days, MACE) ~", paste(pca_features, collapse = " + "))
)
print(surv_formula)

cat("Using core count:", parallel::detectCores(), "\n")
cat("Tuning RSF hyperparameters on training data...\n")
tune_result <- tune.rfsrc(
  surv_formula, 
  data = train, 
  ntreeTry = 300,                     # Fewer trees for speed during tuning
  mtryStart = floor(sqrt(length(pca_features))),  # Starting mtry value
  nodesizeTry = 100,                  # Starting nodesize (adjustable for large datasets)
  doBest = FALSE,                      # Do not automatically refit best model here
  nthread = parallel::detectCores()  # Use all available CPU cores

)
print(tune_result)

# Extract optimal parameters from the tuning result
optimal_mtry <- as.numeric(tune_result$optimal["mtry"])
optimal_nodesize <- as.numeric(tune_result$optimal["nodesize"])
cat("Optimal mtry:", optimal_mtry, "\n")
cat("Optimal nodesize:", optimal_nodesize, "\n")

# Fit the final RSF model on the training set using the tuned parameters
cat("Fitting final RSF model on training data...\n")
rf_tuned <- rfsrc(
  surv_formula, 
  data = train, 
  ntree = 150,          # Increase number of trees for final model
  mtry = optimal_mtry,
  nodesize = optimal_nodesize,
  importance = TRUE,
  do.trace = 100,         # Verbose progress every 100 trees
  nthread = parallel::detectCores()  # Use all available CPU cores
)
print(rf_tuned)

# Evaluate performance on the test set
cat("Predicting on test data...\n")
pred_test <- predict(rf_tuned, newdata = test)
# For each test observation, extract a risk score based on the cumulative hazard function
# Here we use the last value of the cumulative hazard as a simple risk indicator.
risk_score <- sapply(pred_test$chf, function(chf) tail(chf, n = 1))
# Build a survival object for the test set
test_surv <- with(test, Surv(MACE_days, MACE))
# Compute the concordance index using survConcordance (a higher C-index means better predictive performance)
concordance_result <- survConcordance(test_surv ~ risk_score)
c_index <- concordance_result$concordance
cat("Concordance index on test set:", c_index, "\n")

# Plot variable importance from the final RSF model
cat("Plotting variable importance...\n")
var_imp <- rf_tuned$importance
var_imp_df <- data.frame(Feature = names(var_imp), Importance = var_imp)

p <- ggplot(var_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance from RSF Model",
       x = "Feature", y = "Importance")
# Save the plot as a PNG file
ggsave("var_imp_plot.png", plot = p, width = 8, height = 6)
cat("Variable importance plot saved as 'var_imp_plot.png'.\n")

cat("Script execution completed.\n")
