import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb

# Expecting 4 arguments:
# 1) Training dataset (Raw)
# 2) Testing dataset (Raw)
# 3) Existing file to update
# 4) Output file path
if len(sys.argv) != 5:
    print("Usage: python your_script.py <training_input_path> <testing_input_path> <existing_file_path> <output_file_path>")
    sys.exit(1)

training_input_path = sys.argv[1]
testing_input_path = sys.argv[2]
existing_file_path = sys.argv[3]
output_file_path = sys.argv[4]

# Load data
df = pd.read_csv(training_input_path)

# Convert 'year_of_birth' to datetime and calculate 'age'
df = df.copy()
current_date = pd.to_datetime('today')
df['year_of_birth'] = pd.to_datetime(df['year_of_birth'], errors='coerce')
df['age'] = df['year_of_birth'].apply(
    lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day)) if pd.notnull(x) else np.nan
)

# Exclude unwanted columns
exclude = [
    'specimen_id', 'actual_day_relative_to_boost', 'planned_day_relative_to_boost',
    'specimen_type', 'visit', 'ethnicity', 'year_of_birth', 'date_of_boost',
    'dataset', 'timepoint'
]

# Filter datasets
train_df = df[df['dataset'].isin(['2020_dataset', '2021_dataset'])]
test_df = df[df['dataset'] == '2022_dataset']

# Separate day 0 and day 14 data for training
day_0_train = train_df[train_df['planned_day_relative_to_boost'] == 0].set_index('subject_id')
day_14_train = train_df[train_df['planned_day_relative_to_boost'] == 14].set_index('subject_id')

# Separate day 0 and day 14 data for testing
day_0_test = test_df[test_df['planned_day_relative_to_boost'] == 0].set_index('subject_id')
day_14_test = test_df[test_df['planned_day_relative_to_boost'] == 14].set_index('subject_id')

# Ensure the index for day_0 and day_14 data is the same for training
common_indices_train = day_0_train.index.intersection(day_14_train.index)
day_0_train = day_0_train.loc[common_indices_train]
day_14_train = day_14_train.loc[common_indices_train]

# Ensure the index for day_0 and day_14 data is the same for testing
common_indices_test = day_0_test.index.intersection(day_14_test.index)
day_0_test = day_0_test.loc[common_indices_test]
day_14_test = day_14_test.loc[common_indices_test]

# Exclude unwanted columns from both training and testing data
day_0_train = day_0_train.drop(columns=exclude, errors='ignore')
day_0_test = day_0_test.drop(columns=exclude, errors='ignore')

# Prepare features and target variable for training
X_train = day_0_train.copy()
X_train = pd.get_dummies(X_train, columns=['biological_sex', 'race', 'infancy_vac'], drop_first=True)
y_train = day_14_train['IgG_PT']

# Prepare features and target variable for testing
X_test = day_0_test.copy()
X_test = pd.get_dummies(X_test, columns=['biological_sex', 'race', 'infancy_vac'], drop_first=True)
y_test = day_14_test['IgG_PT']

# Align columns in X_test with X_train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Handle missing values
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index].dropna()

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index].dropna()


# Apply log transformation
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

X_train_processed = X_train
X_test_processed = X_test

# Convert to NumPy arrays
X_train_selected = X_train_processed.values
X_test_selected = X_test_processed.values

# Define the models with specified hyperparameters
models = {
    'random_forest': RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ),
    'xgboost': xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.3,
        reg_alpha=0,
        reg_lambda=1,
        objective='reg:squarederror',
        random_state=42
    ),
    'ridge': Ridge(
        alpha=100.0,
        solver='lsqr',
        tol=0.001
    )
}

# Dictionary to store model predictions and performance metrics
model_predictions = {}
model_metrics = {}

# Evaluate each model
for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")
    # Fit the model
    model.fit(X_train_selected, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test_selected)
    predictions_exp = np.expm1(predictions)  # Inverse log transformation
    y_test_exp = np.expm1(y_test)
    
    # Calculate Spearman's rank correlation
    spearman_corr = spearmanr(y_test_exp, predictions_exp).correlation
    mae = mean_absolute_error(y_test_exp, predictions_exp)
    rmse = np.sqrt(mean_squared_error(y_test_exp, predictions_exp))
    
    model_metrics[name] = {
        'Spearman Correlation': spearman_corr,
        'MAE': mae,
        'RMSE': rmse
    }
    
    print(f"Test Spearman's Rank Correlation for {name}: {spearman_corr}")
    print(f"Test MAE for {name}: {mae}")
    print(f"Test RMSE for {name}: {rmse}")
    
    # Store predictions for plotting
    model_predictions[name] = (y_test_exp, predictions_exp)
        
# Print model performance
print("\nModel Performance on Test Set:")
for name, metrics in model_metrics.items():
    print(f"{name}:")
    print(f"  Spearman Correlation: {metrics['Spearman Correlation']}")
    print(f"  MAE: {metrics['MAE']}")
    print(f"  RMSE: {metrics['RMSE']}\n")

# Automatically select the best model based on Spearman Correlation
best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['Spearman Correlation'])
best_model = models[best_model_name]
print(f"The best model is {best_model_name} with Spearman Correlation {model_metrics[best_model_name]['Spearman Correlation']}")

# Now, proceed to use the best model for the unseen dataset

# Step 1: Load and Preprocess the Unseen Dataset
new_df = pd.read_csv(testing_input_path)  # Using sys.argv for the testing dataset path

# Convert 'year_of_birth' to datetime and calculate 'age' if necessary
new_df['year_of_birth'] = pd.to_datetime(new_df['year_of_birth'], errors='coerce')
current_date = pd.to_datetime('today')
new_df['age'] = new_df['year_of_birth'].apply(
    lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day)) if pd.notnull(x) else np.nan
)

exclude = [
    'specimen_id', 'actual_day_relative_to_boost', 'planned_day_relative_to_boost',
    'specimen_type', 'visit', 'ethnicity', 'year_of_birth', 'date_of_boost',
    'dataset', 'timepoint'
]

day_0_new = new_df[new_df['planned_day_relative_to_boost'] == 0].set_index('subject_id')
day_0_new = day_0_new.drop(columns=exclude, errors='ignore')

X_new = day_0_new.copy()
X_new = pd.get_dummies(X_new, columns=['biological_sex', 'race', 'infancy_vac'], drop_first=True)

# Align columns with X_train
X_new = X_new.reindex(columns=X_train.columns, fill_value=0)

missing_values = X_new.isnull().sum()

X_new = X_new.fillna(X_new.mean())

new_indices = X_new.index
X_new_selected = X_new.values

predictions_log = best_model.predict(X_new_selected)
predicted_monocytes_day1 = np.expm1(predictions_log)

monocytes_day_0 = day_0_new.loc[new_indices]['IgG_PT']
monocytes_day_0 = monocytes_day_0.fillna(monocytes_day_0.mean())

monocytes_day_0_log = np.log1p(monocytes_day_0)

fold_change = (predicted_monocytes_day1 + 1) / (monocytes_day_0 + 1)

results_df = pd.DataFrame({
    'subject_id': new_indices,
    'Predicted_Monocytes_Day1': predicted_monocytes_day1,
    'Actual_Monocytes_Day0': monocytes_day_0,
    'Fold_Change': fold_change
})

results_df['Predicted_Monocytes_Rank'] = results_df['Predicted_Monocytes_Day1'].rank(ascending=False)
results_df['Fold_Change_Rank'] = results_df['Fold_Change'].rank(ascending=False)

results_df = results_df.sort_values('Predicted_Monocytes_Rank').reset_index(drop=True)

# Step 1: Load the Existing File
existing_df = pd.read_csv(existing_file_path, delimiter='\t')

results_df = results_df.rename(columns={'subject_id': 'SubjectID'})

update_columns = ['SubjectID', 'Predicted_Monocytes_Rank', 'Fold_Change_Rank']
results_to_merge = results_df[update_columns]
results_to_merge = results_to_merge.rename(columns={
    'Predicted_Monocytes_Rank': '1.1) IgG-PT-D14-titer-Rank',
    'Fold_Change_Rank': '1.2) IgG-PT-D14-FC-Rank'
})

updated_df = existing_df.merge(results_to_merge, on='SubjectID', how='left')

if '1.1) IgG-PT-D14-titer-Rank_x' in updated_df.columns:
    updated_df['1.1) IgG-PT-D14-titer-Rank'] = updated_df['1.1) IgG-PT-D14-titer-Rank_y']
    updated_df.drop(columns=['1.1) IgG-PT-D14-titer-Rank_x', '1.1) IgG-PT-D14-titer-Rank_y'], inplace=True)

if '1.2) IgG-PT-D14-FC-Rank_x' in updated_df.columns:
    updated_df['1.2) IgG-PT-D14-FC-Rank'] = updated_df['1.2) IgG-PT-D14-FC-Rank_y']
    updated_df.drop(columns=['1.2) IgG-PT-D14-FC-Rank_x', '1.2) IgG-PT-D14-FC-Rank_y'], inplace=True)

updated_df.to_csv(output_file_path, sep='\t', index=False)
