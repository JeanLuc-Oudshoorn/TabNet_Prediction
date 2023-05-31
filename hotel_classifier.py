# Imports
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
import os

# Print CUDA compatible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = f"1"

# Print PyTorch version
print(torch.__version__)

# Silence future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Read in data
train = pd.read_csv('bookings_train.csv')
test = pd.read_csv('bookings_test_solutions.csv')

# Denote train, test and validation observations
train['Set'] = np.random.choice(['train', 'valid'], p=[.85, .15], size=(train.shape[0],))
test['Set'] = 'test'

# Merge all observations into a single dataframe for preprocessing
train = pd.concat([train, test])

# Convert outcome to binary
target = 'is_cancelled'
train['is_cancelled'] = np.multiply(train['is_cancelled'] == 'yes', 1)

# Denote indices
train_indices = train[train.Set == "train"].index
valid_indices = train[train.Set == "valid"].index
test_indices = train[train.Set == "test"].index

###############################################################################
#                  1. Feature Engineering & Preprocessing                     #
###############################################################################

print("\n Start Preprocessing")

# Feature Engineering
train.rename(columns={'arrival_date_day_of_month': 'day',
                     'arrival_date_year': 'year',
                     'arrival_date_month': 'month',
                     'arrival_date_week_number': 'weeknum'},
            inplace=True)

# Convert month to number
train['month'] = train['month'].apply(lambda x: datetime.strptime(x, "%B").month)

# Create date
train['date'] = pd.to_datetime(train[['year', 'month', 'day']],
                              format="%Y%B%d")

# Extract day of week
train['weekday'] = train['date'].dt.dayofweek

# Binary: Customer got reserved room
train['got_reserved_room'] = np.multiply((train['reserved_room_type'] == train['assigned_room_type']), 1)

# Total visitors
train['total_visitors'] = train['adults'] + train['children'] + train['babies']

# Check for missing values
np.sum(train.isna())
train['country'] = train['country'].fillna(value='Other')

# Label encode categorical features and fill empty cells
nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims = {}

for col in train.columns:
    if types[col] == 'object' or nunique[col] < 25:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

print("\n Preprocessing Done")

###############################################################################
#                      2. Define Categorical Variables                        #
###############################################################################

unused_feat = ['Set', 'date']

features = [col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

###############################################################################
#                        3. Set up TabNet Classifier                          #
###############################################################################

tabnet_params = {"cat_idxs": cat_idxs,
                 "cat_dims": cat_dims,
                 "cat_emb_dim": 4,
                 "optimizer_fn": torch.optim.Adam,
                 "optimizer_params": dict(lr=2e-2),
                 "scheduler_params": {"step_size": 50, "gamma": 0.9},
                 "scheduler_fn": torch.optim.lr_scheduler.StepLR,
                 "mask_type": 'sparsemax'
                }

clf = TabNetClassifier(**tabnet_params)

# Set max epochs
max_epochs = 80

# Define training and test data
X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

###############################################################################
#                           4. Fit the Classifier                             #
###############################################################################

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=max_epochs, patience=25,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    augmentations=None
)

# Plot training progress
fig, ax = plt.subplots(1, 2)
ax[0].plot(clf.history['loss'])
ax[0].set_title('Loss')
ax[1].plot(clf.history['train_auc'])
ax[1].plot(clf.history['valid_auc'])
ax[1].set_title('Train- & Val. AUC')
plt.savefig('training_progress.png')
plt.show()

###############################################################################
#                             5. Make Predictions                             #
###############################################################################

preds = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)


preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)

print(f"BEST VALID SCORE FOR HOTEL CANCELLATION DATA : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR HOTEL CANCELLATION DATA : {test_auc}")

# Check that best weights are used
assert np.isclose(valid_auc, np.max(clf.history['valid_auc']), atol=1e-6)

###############################################################################
#                              6. Save the Model                              #
###############################################################################

# Save tabnet model
saving_path_name = "./tabnet_hotel_model"
saved_filepath = clf.save_model(saving_path_name)

# Define new model with basic parameters and load state dict weights
loaded_clf = TabNetClassifier()
loaded_clf.load_model(saved_filepath)

# Check that the loaded model behaves the same way
loaded_preds = loaded_clf.predict_proba(X_test)
loaded_test_auc = roc_auc_score(y_score=loaded_preds[:, 1], y_true=y_test)

print(f"FINAL TEST SCORE FOR HOTEL CANCELLATION DATA : {loaded_test_auc}")

assert(test_auc == loaded_test_auc)

loaded_clf.predict(X_test)

###############################################################################
#                           7. Feature Importance                             #
###############################################################################

# Importances sum up to one
print(clf.feature_importances_)

# Extract explainability matrix and masks
explain_matrix, masks = clf.explain(X_test)

# Plot global feature importance
_, axs = plt.subplots(1, 3, figsize=(20, 20))

for i in range(3):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f"mask {i}")
    axs[i].set_xticklabels(labels=features, rotation=45)

plt.savefig('feature_importance.png')
plt.show()
