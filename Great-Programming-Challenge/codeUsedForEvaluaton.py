import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
# from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
import xgboost as xgb



# def partition(data, fraction):
#     ldata = list(data)
#     breakPoint = int(len(ldata) * fraction)
#     return ldata[:breakPoint], ldata[breakPoint:]






print("######### START ###########")

print("\n   #### INPUT TREATEMENT ####  ")

file_path_train = 'TrainOnMe.csv'

inputs = []
classes = []


columns_used = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x13']
columns_used_evaluate = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x13']
classes_used = ['Atsutobob', 'Boborg', 'Jorgsuto']
classes_used_int = [0, 1, 2]
inputs = pd.read_csv(file_path_train, usecols=columns_used)

bad_inputs = inputs[~inputs['y'].isin(classes_used)]
inputs.drop(bad_inputs.index, inplace=True)
inputs['x1'] = inputs['x1'].astype(float)
bad_inputs = inputs[inputs['x1'] > 110]
inputs.drop(bad_inputs.index, inplace=True)

inputs['y'].replace(classes_used[0], classes_used_int[0], inplace=True)
inputs['y'].replace(classes_used[1], classes_used_int[1], inplace=True)
inputs['y'].replace(classes_used[2], classes_used_int[2], inplace=True)
inputs['y'] = inputs['y'].astype(int)
inputs['x7'].replace('Polkagriss', 'Polskorgris', inplace=True)
inputs['x7'].replace('Schottisgriss', 'Schottisgris', inplace=True)   
inputs =pd.get_dummies(inputs)
features_used = inputs.columns
features_used = features_used[1:]


nan_inputs = inputs[inputs.isna().any(axis=1)]
inputs.drop(nan_inputs.index, inplace=True)

# print("--df")
# print(inputs)
# print("--types")
# print(inputs.dtypes)




# print("\n-------------------------------------------------------\n")

file_path_evaluate = 'EvaluateOnMe.csv'
file_path_solution = 'Solution.txt'

inputs_evaluate = pd.read_csv(file_path_evaluate, usecols=columns_used_evaluate)
inputs_evaluate['x1'] = inputs_evaluate['x1'].astype(float)
inputs_evaluate =pd.get_dummies(inputs_evaluate)
features_used_evaluate = inputs_evaluate.columns
# print(inputs_evaluate)
print(f"\nshape of dataframe of solution: {inputs_evaluate.shape}")





searching_hyperparameter = False #TODO : to modify depending if searching hyperparamter or compute solution








# modify the hyperparameter you want to study
if searching_hyperparameter:
    print("\n   #### SEARCHING HYPERPARAMETER ####  ")
    hyper_params = [8] # here differents values of hyperparameter you want to try
    hyp_score = []

    for hyper_param in hyper_params:
        nb_estimation = 100
        score = []
        for estimation in range(nb_estimation):
            # np.random.seed(42)
            fraction = 0.7
            random_indices = np.random.rand(len(inputs)) < fraction
            inputs_train = inputs[random_indices]
            inputs_test = inputs[~random_indices]

            clf = RandomForestClassifier(random_state=42,min_samples_split=2, bootstrap= False , criterion='entropy', max_features='log2', n_estimators=100, max_depth=8)

            boost_clf = AdaBoostClassifier(clf, n_estimators= 100, learning_rate=0.01)
            boost_clf = boost_clf.fit(inputs_train.loc[:, features_used], inputs_train.loc[:, 'y'])

            result = boost_clf.predict(inputs_test.loc[:, features_used])

            score.append(0)
            for i in range(len(result)):
                if result[i]==inputs_test['y'].values[i]:
                    score[estimation] +=1
            score[estimation] = score[estimation]/len(result)
            print(f"score iteration {estimation}: {score[estimation]}")
        hyp_score.append(np.mean(score))
        print(f"----mean score for hyper_param={hyper_param}:")
        print(np.mean(score))
        print("\n")

    plt.plot(hyper_params, hyp_score, label='label')
    plt.title('Effect of parameter')
    plt.xlabel('values of parameter')
    plt.ylabel('score')
    plt.tight_layout()
    plt.show()



if not searching_hyperparameter:
    print("\n   #### EVALUATION TREATEMENT ####")

    print("\n--Training model...")

    clf = RandomForestClassifier(random_state=42,min_samples_split=2, bootstrap= False , criterion='entropy', max_features='log2', n_estimators=1000, max_depth=8)
    boost_clf = AdaBoostClassifier(clf, n_estimators= 1000, learning_rate=0.01)
    boost_clf = boost_clf.fit(inputs.loc[:, features_used], inputs.loc[:, 'y'])


    print("\n--Predict solution...")

    file_path_solution = './Solution.txt'

    result = boost_clf.predict(inputs_evaluate.loc[:, features_used_evaluate])
    # print(result)
    print(f"\nshape of dataframe to evaluate: {result.shape}")

    print("\n--Printing solution on Solution.txt...\n")
    results_labelled = []
    for res in result:
        if res==classes_used_int[0]:
            results_labelled.append(classes_used[0])
        elif res==classes_used_int[1]:
            results_labelled.append(classes_used[1])
        elif res==classes_used_int[2]:
            results_labelled.append(classes_used[2])
        else:
            results_labelled.append('ERROR_LABEL_NOT_FIND')
    
    with open(file_path_solution, 'w') as file:
        # Write each row of data
        for row in results_labelled:
            file.write(f"{row}\n")

    print(f"{results_labelled[:10]}...")
    
print("\n   #### END PROGRAMME ####  ")

print("\n########## END ############")