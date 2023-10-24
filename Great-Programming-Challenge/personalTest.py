# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# # Values to exclude
# exclude_values = [3, 7]

# # Create a slice that includes all elements except the specified values
# result = [value for value in my_list if value not in exclude_values]

# print(result)












from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

# Load the Iris dataset as an example
iris = load_iris()
X, y = iris.data, iris.target

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [ 'sqrt', 'log2']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)

# Get the best parameters
best_params = grid_search.best_params_

# Get the best model
best_rf_model = grid_search.best_estimator_
