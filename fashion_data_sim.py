# Imports
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Create empty dataframe
data = pd.DataFrame(np.ones((130000, 21)))

# Determine number of samples
n_samples = 130000

# Create dataset
X, y = make_classification(n_samples=n_samples, n_features=9, n_informative=9, n_redundant=0, n_classes=2,
                           n_clusters_per_class=10, class_sep=1.0, flip_y=0.05, weights=[0.72, 0.28])

# Add outcome to the dataset
data[14] = y

X = pd.DataFrame(X)

# Validating importances
# --------------------------------------------------------------------------------------- #

# Create data splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Verify accuracy
print(clf.score(X_test, y_test))

# Check for most important features
result = permutation_importance(clf, X_test, y_test, n_repeats=4, random_state=42)
print(result['importances_mean'])

# --------------------------------------------------------------------------------------- #

# Normalize all features
X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Check distributions
print(X.describe())

X = X.values

# Color discretizer
col_est = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='uniform')
data[0] = col_est.fit_transform(X[:, 2].reshape(-1, 1))

# Brand discretizer
br_est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data[1] = br_est.fit_transform(X[:, 4].reshape(-1, 1))

# Order ID
data[2] = np.random.choice(np.arange(125000), size=130000, replace=True)

data.sort_values(2, inplace=True)

# Customer ID
data[3] = data[2] - data[2].shift(-1)
data[3] = data[3].shift(1)

data[4] = np.random.choice(np.arange(60000), size=130000, replace=True)
data[4] = np.where(data[3] == 0, np.nan, data[4])
data[4] = data[4].ffill()

# Check order distribution per customer ID
data.groupby(4)[2].nunique().value_counts()

# Create Product group 1 ID
pg_est = KBinsDiscretizer(n_bins=18, encode='ordinal', strategy='uniform')
data[5] = pg_est.fit_transform(X[:, 0].reshape(-1, 1))

# Create Product cat ID
pc_est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
data[6] = pc_est.fit_transform(data[5].values.reshape(-1, 1))

# Size discretizer
size_est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
data[7] = size_est.fit_transform(X[:, 3].reshape(-1, 1))

# Sold to country discretizer
scountry_est = KBinsDiscretizer(n_bins=14, encode='ordinal', strategy='uniform')
data[8] = scountry_est.fit_transform(X[:, 1].reshape(-1, 1))

# Prod. country discretizer
pcountry_est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
data[9] = pcountry_est.fit_transform(X[:, 5].reshape(-1, 1))

# Fit discretizer
fit_est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
data[10] = fit_est.fit_transform(X[:, 6].reshape(-1, 1))

# Price
data[11] = np.round(X[:, 8] * 140 + np.random.randint(low=2, high=160, size=130000)) \
           - np.random.choice([-100, -50, 0], size=130000, replace=True, p=[0.6, 0.2, 0.2])

# Month discretizer
month_est = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='uniform')
data[12] = month_est.fit_transform(X[:, 7].reshape(-1, 1)) + 1

# Day of month discretizer
long_months = [1, 3, 5, 7, 8, 10, 12]
data[13] = np.where(data[12].astype(int).isin(long_months),
                                        np.random.choice(np.arange(1, 32), size=130000),
                                        np.random.choice(np.arange(1, 31), size=130000))

# Drop obsolete columns
data.drop(3, axis=1, inplace=True)
data = data.iloc[:, 0:14]

# Assign column names
data.columns = ['color', 'brand', 'order_id', 'customer_id', 'product_group', 'product_class', 'size',
                'sold_to_country', 'prod_country', 'fit', 'price', 'month', 'day', 'is_returned']

# # Write CSV
# data.to_csv('fashion_data.csv', index=False)
#
# # Read the data
# data = pd.read_csv('fashion_data.csv')

# Int conversion
data = data.astype(int)

# Update month
data['month'] = np.where(data['month'] == 7, np.random.choice([7, 8], size=130000, p=[0.7, 0.3]), data['month'])
data['month'] = np.where(data['month'] == 6, np.random.choice([6, 7], size=130000, p=[0.75, 0.25]), data['month'])
data['month'] = np.where(data['month'] == 8, np.random.choice([8, 9], size=130000, p=[0.85, 0.15]), data['month'])

data['month'].value_counts()

# Fix february dates
data['day'] = np.where(data['month'] == 2, np.random.choice(np.arange(1, 29)), data['day'] )

# Year
data['year'] = np.where(data['month'].isin([8, 9]), np.random.choice([2021, 2022], size=130000, p=[0.5, 0.5]), 2021)
data['year'] = np.where(data['month'].isin([10, 11, 12]), 2021, data['year'])
data['year'] = np.where(data['month'].isin([1, 2, 3, 4, 5, 6, 7]), 2022, data['year'])

# Create order date
data['order_date'] = pd.to_datetime(data[['year', 'month', 'day']], errors='coerce')

# Fix order dates same time
data['order_id_ps'] = data['order_id'] - data['order_id'].shift(-1)
data['order_id_ps'] = data['order_id_ps'].shift(1)

data['order_date'] = np.where(data['order_id_ps'] == 0, np.datetime64("NaT"), data['order_date'])
data['order_date'] = data['order_date'].ffill()

# Drop missing observations from time conversion
data.dropna(inplace=True)

# Sort by order date
data.sort_values(['order_date', 'order_id'], inplace=True)

# Cust features
data['cust_ret'] = data.groupby('customer_id')['is_returned'].transform(lambda x: x.expanding(0).sum())
data['cust_pur'] = data.groupby('customer_id')['is_returned'].transform(lambda x: x.expanding(0).count())

data['pc_cust_ret'] = data.groupby(['customer_id', 'product_class'])['is_returned'].transform(lambda x: x.expanding(0).sum())
data['pc_cust_pur'] = data.groupby(['customer_id', 'product_class'])['is_returned'].transform(lambda x: x.expanding(0).count())

# Fix zero
data['cust_ret'] = np.where(data['cust_ret'] < 0, 0, data['cust_ret'])
data['cust_pur'] = np.where(data['cust_pur'] < 0, 0, data['cust_pur'])
data['pc_cust_ret'] = np.where(data['pc_cust_ret'] < 0, 0, data['pc_cust_ret'])
data['pc_cust_pur'] = np.where(data['pc_cust_pur'] < 0, 0, data['pc_cust_pur'])

# Adjust outcome based on customer features
data['is_returned_ps'] = np.where(data['cust_ret'] > 0, np.random.choice([1, 2], size=len(data), p=[0.8, 0.2]), 1)
data['is_returned_ps'] = np.where(data['cust_ret'] > 1, np.random.choice([1, 2], size=len(data), p=[0.75, 0.25]), data['is_returned_ps'])

data['is_returned_ps'] = np.where(data['cust_pur'] > 2, np.random.choice([1, 3], size=len(data), p=[0.95, 0.05]), data['is_returned_ps'])
data['is_returned_ps'] = np.where(data['cust_pur'] > 4, np.random.choice([1, 3], size=len(data), p=[0.95, 0.05]), data['is_returned_ps'])

data['is_returned_ps'] = np.where(data['pc_cust_ret'] > 0, np.random.choice([1, 2], size=len(data), p=[0.8, 0.2]), data['is_returned_ps'])
data['is_returned_ps'] = np.where(data['pc_cust_ret'] > 1, np.random.choice([1, 2], size=len(data), p=[0.7, 0.3]), data['is_returned_ps'])

# Order features
n_item_pc = pd.DataFrame(data.groupby(['order_id', 'product_class'])['is_returned'].count())
n_item = data.groupby('order_id')['is_returned'].count()

n_item_pc['order_id_new'] = n_item_pc.index.get_level_values(0)
n_item_pc['product_class_new'] = n_item_pc.index.get_level_values(1)

data = data.merge(n_item, left_on=data['order_id'], right_on=n_item.index, how='left')
data = data.merge(n_item_pc, left_on=['key_0', 'product_class'], right_on=['order_id_new', 'product_class_new'],
                  how='left')

data.rename(columns={'is_returned': 'n_item_pc', 'is_returned_y': 'n_item'}, inplace=True)
data.rename(columns={'is_returned_x': 'is_returned'}, inplace=True)

data.drop(['key_0', 'order_id_new', 'product_class_new'], axis=1, inplace=True)

data['is_returned_ps'] = np.where(data['n_item'] > 1, np.random.choice([1, 4], size=len(data), p=[0.9, 0.1]), data['is_returned_ps'])
data['is_returned_ps'] = np.where(data['n_item'] > 3, np.random.choice([1, 4], size=len(data), p=[0.85, 0.15]), data['is_returned_ps'])

data['is_returned_ps'] = np.where(data['n_item_pc'] > 1, np.random.choice([1, 4], size=len(data), p=[0.9, 0.1]), data['is_returned_ps'])
data['is_returned_ps'] = np.where(data['n_item_pc'] > 2, np.random.choice([1, 4], size=len(data), p=[0.85, 0.15]), data['is_returned_ps'])

print(data['is_returned_ps'].value_counts())

# Fix return status
data['draw1'] = np.random.choice([0, 1], size=len(data), p=[0.45, 0.55])
data['draw2'] = np.random.choice([0, 1], size=len(data), p=[0.88, 0.12])
data['draw3'] = np.random.choice([0, 1], size=len(data), p=[0.4, 0.6])

data['is_returned'] = np.where(data['is_returned_ps'] == 2, data['draw1'], data['is_returned'])
data['is_returned'] = np.where(data['is_returned_ps'] == 3, data['draw2'], data['is_returned'])
data['is_returned'] = np.where(data['is_returned_ps'] == 4, data['draw3'], data['is_returned'])

# Check if the transformation was successful
print(data.groupby('cust_ret')['is_returned'].mean())
print(data.groupby('cust_pur')['is_returned'].mean())
print(data.groupby('pc_cust_ret')['is_returned'].mean())
print(data.groupby('pc_cust_pur')['is_returned'].mean())

print(data.groupby('n_item_pc')['is_returned'].mean())
print(data.groupby('n_item')['is_returned'].mean())

# Drop obsolete columns
data.drop(columns=['is_returned_ps', 'draw1', 'draw2', 'draw3'], inplace=True)

# Add days after order
data['days_after_order'] = np.where((data['is_returned'] == 1) & (data['sold_to_country'] <= 7),
                                    np.random.poisson(9, len(data)), np.nan)

data['days_after_order'] = np.where((data['is_returned'] == 1) & (data['sold_to_country'] > 7),
                                    np.random.poisson(13, len(data)), data['days_after_order'])

data['days_after_order'] = data['days_after_order'] + 1

# Return date
data['return_date'] = data['order_date'] + pd.TimedeltaIndex(data['days_after_order'], unit='D')

# Change categorical variables
cat_list = ['color', 'brand', 'product_group', 'product_class', 'size', 'sold_to_country', 'prod_country',
            'fit', 'month', 'year']

num_list = ['price', 'cust_ret', 'cust_pur', 'pc_cust_ret', 'pc_cust_pur', 'n_item_pc', 'n_item']

pred_list = cat_list + num_list

for i in cat_list:
    data[i] = data[i].astype('category')

# Split data
X_train_w = data[data['order_date'] <= '2022-07-15']
X_test_w = data[(data['order_date'] >= '2022-08-15') & (data['order_date'] <= '2022-09-15')]

y_train = X_train_w['is_returned']
y_test = X_test_w['is_returned']

X_train = X_train_w.drop(np.setdiff1d(X_train_w.columns, pred_list), axis=1)
X_test = X_test_w.drop(np.setdiff1d(X_test_w.columns, pred_list), axis=1)

# Fit lightgbm
lgbm_estimator = lgb.LGBMClassifier(n_jobs=-1)

lgbm_estimator.fit(X_train, y_train)

print(lgbm_estimator.score(X_test, y_test))

# Check for most important features
result = permutation_importance(lgbm_estimator, X_test, y_test, n_repeats=5, random_state=42)
print(dict(zip(result['importances_mean'], X_test.columns)))

importances = pd.DataFrame.from_dict(dict(zip(X_test.columns, result['importances_mean'])), orient='index')
importances.plot.barh()

# Mapping color
col_list = ['yellow', 'orange', 'brown', 'black', 'white', 'blue', 'red', 'beige', 'multi', 'green', 'violet', 'unknown']
color_dict = dict(zip(np.arange(0, 12), col_list))

data['color'] = data['color'].map(color_dict)

# Mapping brand
br_list = ['Pull & Bear', 'Zara', 'Massimo Dutti']
br_dict = dict(zip(np.arange(0, 3), br_list))

data['brand'] = data['brand'].map(br_dict)

# Mapping product class
pcat_list = ['Formal', 'Denim', 'Non-denim', 'Accessories', 'Casual', 'Footwear']
pcat_dict = dict(zip(np.arange(0, 5), pcat_list))

data['product_class'] = data['product_class'].map(pcat_dict)

# Mapping product group
pg_list = ['Shirt', 'Trousers', 'Suit', 'Jeans', 'Jacket', 'Overall', 'Tee', 'Chino', 'Dress', 'Scarf', 'Handbag',
           'Belt', 'Pants', 'Polo', 'Shirt', 'Loafer', 'Boot', 'Sneak']
pg_dict = dict(zip(np.arange(0, 18), pg_list))

data['product_group'] = data['product_group'].map(pg_dict)

# Mapping size
sz_list = ['XXS', 'XL', 'M', 'L', 'S', 'XS', 'XXL']
sz_dict = dict(zip(np.arange(0, 6), sz_list))

data['size'] = data['size'].map(sz_dict)

# S-Country
sc_list = ['Estonia', 'Austria', 'Finland', 'Sweden', 'Denmark', 'Great Britain', 'Germany', 'Spain', 'France', 'Italy',
           'Switzerland', 'Portugal', 'Netherlands', 'Belgium']
sc_dict = dict(zip(np.arange(0, 14), sc_list))

data['sold_to_country'] = data['sold_to_country'].map(sc_dict)

# P-Country
pc_list = ['China', 'Vietnam', 'Bangladesh', 'Portugal', 'Turkey', 'India']
pc_dict = dict(zip(np.arange(0, 6), pc_list))

data['prod_country'] = data['prod_country'].map(pc_dict)

# Fit
fit_list = ['Skinny', 'Slim', 'Regular', 'Comfort']
fit_dict = dict(zip(np.arange(0, 6), fit_list))

data['fit'] = data['fit'].map(fit_dict)

# Fix size and fits
data['fit'] = np.where(data['product_class'] == 'Accessories', np.nan, data['fit'])
data['size'] = np.where(data['product_class'] == 'Accessories', np.nan, data['size'])

data['size'] = np.where(data['product_class'] == 'Footwear',  np.random.choice(np.arange(37, 46),
                                                                                      size=len(data)), data['size'])

# Some more things
data['is_returned'] = np.where(data['sold_to_country'] == 'Germany', np.random.choice([0, 1], size=len(data), p=[0.54, 0.46]), data['is_returned'])
data['is_returned'] = np.where(data['sold_to_country'] == 'Switzerland', np.random.choice([0, 1], size=len(data), p=[0.6, 0.4]), data['is_returned'])

data['is_returned'] = np.where(data['product_group'] == 'Handbag', np.random.choice([0, 1], size=len(data), p=[0.9, 0.1]), data['is_returned'])
data['is_returned'] = np.where(data['product_group'] == 'Scarf', np.random.choice([0, 1], size=len(data), p=[0.85, 0.15]), data['is_returned'])
data['is_returned'] = np.where(data['product_group'] == 'Dress', np.random.choice([0, 1], size=len(data), p=[0.55, 0.45]), data['is_returned'])

data['is_returned'] = np.where(data['prod_country'] == 'Vietnam', np.random.choice([0, 1], size=len(data), p=[0.89, 0.11]), data['is_returned'])

# Drop columns
data.drop(['cust_ret', 'cust_pur', 'pc_cust_ret', 'pc_cust_pur', 'n_item', 'n_item_pc', 'days_after_order'], axis=1, inplace=True)
data.drop(['day'], axis=1, inplace=True)

# Write CSV
data.to_csv('fashion_data.csv', index=False)
