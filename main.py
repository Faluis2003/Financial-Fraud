import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn import model_selection
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

# Function to check for missing data in a DataFrame
def handle_missing_data(data):
    data_null = data.isnull().sum().index[data.isnull().sum() != 0].tolist()
    if len(data_null) == 0:
        return True
    elif len(data_null) != 0:
        for i in data_null:
            if data[i].dtype == 'float64' or data[i].dtype == 'int64':
                data[i].fillna(data[i].median(), inplace=True)
            else:
                pass
        return False

# Read the dataset from a CSV file
Dataset = pd.read_csv('archive/PS_20174392719_1491204439457_log.csv')

# Check for missing data in the dataset
if not handle_missing_data(Dataset):
    print('The file had missing data.')
else:
    # Drop rows with missing data
    Dataset = Dataset.dropna()

# Map 'isFraud' and 'type' columns to meaningful labels
Dataset['isFraud'] = Dataset['isFraud'].map({0: 'Not Fraud', 1: 'Fraud'})
Dataset['type'] = Dataset['type'].map({'PAYMENT': 1, 'TRANSFER': 4, 'CASH_OUT': 2, 'DEBIT': 5, 'CASH_IN': 3})

# Select features (x) and target variable (y)
x = Dataset[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(Dataset['isFraud'])

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

# Create a list of regression models
regression_models = [
    #('Linear Regression', LinearRegression()),
    #('Logistic Regression', LogisticRegression()),
    #('Ridge Regression', Ridge()),
    #('Lasso Regression', Lasso(max_iter=10000)),
    #('Decision Tree Regressor', DecisionTreeRegressor()),
    #('Random Forest Regressor', RandomForestRegressor()),
    #('Gradient Boosting Regressor', GradientBoostingRegressor()),
    #('XGBoost Regressor', XGBRegressor()),
    #('LightGBM Regressor', LGBMRegressor(force_row_wise=True))
]
#Regression Models:
#Linear Regression: -0.000977 (0.000049)
#Logistic Regression: -0.000488 (0.000250)
#Ridge Regression: -0.000977 (0.000049)
#Lasso Regression: -0.000979 (0.000049)
#Decision Tree Regressor: -0.000608 (0.000057)
#Random Forest Regressor: -0.000382 (0.000040)
#Gradient Boosting Regressor: -0.000729 (0.000051)
#XGBoost Regressor: -0.000437 (0.000038)
#LightGBM Regressor: -0.000430 (0.000037)


# Select a subset of features for clustering
clustering_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
x_clustering = Dataset[clustering_features]

# Downcast data types to reduce memory usage
x_clustering = x_clustering.astype('float32')
# Create a list of clustering and association models
# Uncomment the other clustering models for comparison
clustering_models = [
    ('K-Means Clustering',  MiniBatchKMeans(n_clusters=2, random_state=42)),
    ('Hierarchical Clustering', AgglomerativeClustering(linkage='average')),
    ('Gaussian Mixture Model', GaussianMixture(n_components=2)),
]

# Evaluation for clustering models
print("\nClustering Models:")
for name, model in clustering_models:
    model.fit(x_clustering)
    # Evaluate clustering models using silhouette score
    silhouette_score = metrics.silhouette_score(x_clustering, model.labels_)
    print(f"{name}: Silhouette Score = {silhouette_score}")
    print(f"{name}: Labels = {model.labels_}")

# Evaluation for regression models
print("Regression Models:")
for name, model in regression_models:
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Association Rule Mining (Apriori)
# Assuming you have a dataset with transactional data, and you want to mine association rules
transactions_df = pd.get_dummies(Dataset['type'])
frequent_itemsets = apriori(transactions_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
print("\nAssociation Rules:")
print(rules)