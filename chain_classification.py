# Imports
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

n_samples = 10000

# Create dataset
X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=4, n_classes=3)

y = y.reshape(n_samples, 1)

# Create a categorical variable
X[:, 0] = np.random.choice(5, size=n_samples)

feat_names = ["feat_" + str(num) for num in np.arange(20)]
X = pd.DataFrame(X, columns=feat_names)

X["feat_0"] = X["feat_0"].astype("category")

# Create another categorical variable
X["feat_20"] = np.random.choice(3, size=n_samples)

X["feat_20"] = X["feat_20"].astype("category")

#Create multiple outputs
y = np.resize(np.repeat(y, 20), (n_samples, 20))

weights = np.concatenate((np.repeat(0, 6), np.random.randint(0, 3, size=7), np.repeat(1, 7)))

# Create dependency among output classes
y = np.random.permutation(y)

# Create train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base model
model = lgb.LGBMClassifier(n_jobs=-1)

# Create a wrapper for base model
wrapper = ClassifierChain(model, verbose=True, random_state=42)

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
pred_proba = wrapper.predict_proba(X_test)

preds_reshaped = np.reshape(preds, (len(X_test) * 20, 1))
y_test_reshaped = np.reshape(y_test, (len(X_test) * 20, 1))
pred_proba_reshaped = np.reshape(pred_proba, (len(X_test) * 20, 1))

# Evaluate output
side_by_side = list(zip(preds_reshaped, pred_proba_reshaped))

for i in range(200):
    print(side_by_side[i])
