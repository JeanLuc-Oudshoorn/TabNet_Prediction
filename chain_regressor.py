# Imports
import numpy as np
import lightgbm as lgb
import time
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import RegressorChain

# Create dataset
X, y = make_regression(n_samples=1800000, n_features=30, n_informative=11, n_targets=100, random_state=1, noise=0.5)

# Create linear dependency across outputs
multp_vec = np.linspace(1, 2.5, 100)

y = y * multp_vec[np.newaxis, :]

# Create train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define base model
model = lgb.LGBMRegressor(n_jobs=-1)

# Create a wrapper for base model
wrapper = RegressorChain(model, verbose=True, random_state=42)

# Save start time
st = time.time()

# Fit the model to training observations
wrapper.fit(X_train, y_train)

# Save end time
et = time.time()

# Print elapsed time
elapsed_time = et - st
print('Execution time:', np.round(elapsed_time/60, 1), 'minutes')

# Make predictions
preds = wrapper.predict(X_test)

# Evaluate prediction quality
score = mean_absolute_percentage_error(y_true=y_test, y_pred=preds)

# Print evaluation score
print("Mean absolute percentage error:", np.round(score, 2))
