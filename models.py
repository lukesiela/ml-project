import argparse
from enum import auto
import sys
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import sklearn.model_selection  as skms
import sklearn.metrics          as skm

import sklearn.neighbors        as skn
import sklearn.linear_model     as sklm
import sklearn.tree             as skt
import sklearn.svm              as sksvm
import sklearn.ensemble         as ske


# usage 
def usage():
    print('Usage: ./models.py --data <data_filepath> --models <\'rand\', \'prob\', \'knn\', \'mlp\', \'tree\', \'svm\', \'lr\', and/or \'rf\'> (--cross) (--tune)')
    print('Note: use --cross to perform cross-validation instead of single train-test split')
    print('Note: use --tune to perform hyperparameter tuning instead of single train-test split')
    print('Example: ./models.py --data data/preprocessed/battles010121_pp.csv --models rand prob rf')
    exit(1)

# parse arguments
def parse():
    if len(sys.argv) < 5:
        usage()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data')
    parser.add_argument('--models', nargs='+', dest='models')
    parser.add_argument('--cross', action='store_true')
    parser.add_argument('--tune', action='store_true')
    # parser.add_argument('--output', dest='output')
    return parser.parse_args()

# run models
def main():
    args = parse()

    # read data
    print('Reading data...')
    matchups = pd.read_csv(args.data, index_col=0)

    # split into train test if not using cross-validation
    if args.cross:
        train_X = matchups.iloc[:, :-1]
        train_y = matchups.iloc[:, -1]
    else:
        train_X, test_X, train_y, test_y = skms.train_test_split(
            matchups.iloc[:,:-1], 
            matchups.iloc[:,-1], 
            test_size=0.10
        )

    print('Running models...')

    # random choice
    if 'rand' in args.models:
        rand_pred = [random.randint(0, 1) for _ in train_y]
        print('Random Choice: ' + str(skm.accuracy_score(train_y, rand_pred)))

    # player with higher total card levels predicted to win
    if 'prob' in args.models:
        prob_pred = [1 if matchup[1].iloc[:102].sum() > matchup[1].iloc[102:].sum() else 0 for matchup in train_X.iterrows()]
        print('Card Levels: ' + str(skm.accuracy_score(train_y, prob_pred)))
        
    if args.cross:
        print()
        
    # knn with a few values for k
    # none could beat random choice
    # WARNING: takes very long (hours +)
    if 'knn' in args.models:
        if args.cross:

            # cross-validation not feasible
            print('WARNING: skipping KNN model due to computing constraints during cross-validation.  To use this model, run the program without the --cross flag.')
        else:
            for k in tqdm([1,11,21]):
                knn = skn.KNeighborsClassifier(k)
                knn = knn.fit(train_X, train_y)
                knn_pred = knn.predict(test_X)
                print('KNN (' + str(k) + '): ' + str(skm.accuracy_score(test_y, knn_pred)))

    # perceptron
    if 'mlp' in args.models:
        mlp = sklm.Perceptron()
        mlp = mlp.fit(train_X, train_y)
        if args.cross:
            scores = skms.cross_val_score(mlp, train_X, train_y, cv=5)
            print('Perceptron')
            print('Scores: ' + ', '.join([str(score)[:8] for score in scores]))
            print('Mean:   ' + str(np.average(scores)))
            print('StDev:  ' + str(np.std(scores)))
            print()
        else:
            mlp_pred = mlp.predict(test_X)
            print('Perceptron: ' + str(skm.accuracy_score(test_y, mlp_pred)))

    # decision tree
    if 'tree' in args.models:
        dtc = skt.DecisionTreeClassifier()
        dtc = dtc.fit(train_X, train_y)
        if args.cross:
            scores = skms.cross_val_score(dtc, train_X, train_y, cv=5)
            print('Decision Tree')
            print('Scores: ' + ', '.join([str(score)[:8] for score in scores]))
            print('Mean:   ' + str(np.average(scores)))
            print('StDev:  ' + str(np.std(scores)))
            print()
        else:
            dtc_pred = dtc.predict(test_X)
            print('Decision Tree: ' + str(skm.accuracy_score(test_y, dtc_pred)))

    # SVM
    # WARNING: takes very long (hours +)
    if 'svm' in args.models:
        if args.cross:

            # cross-validation not feasible
            print('WARNING: skipping SVM model due to computing constraints during cross-validation.  To use this model, run the program without the --cross flag.')
        else:
            svm = sksvm.SVC(kernel='rbf', decision_function_shape='ovr', C=10.0, tol=0.001, max_iter=-1)
            svm = svm.fit(train_X, train_y)
            svm_pred = svm.predict(test_X)
            print('SVM: ' + str(skm.accuracy_score(test_y, svm_pred)))

    # Logistic Regression
    if 'lr' in args.models:
        lrm = sklm.LogisticRegression(max_iter=10000)
        lrm = lrm.fit(train_X, train_y)
        if args.cross:
            scores = skms.cross_val_score(lrm, train_X, train_y, cv=5)
            print('Logistic Regression')
            print('Scores: ' + ', '.join([str(score)[:8] for score in scores]))
            print('Mean:   ' + str(np.average(scores)))
            print('StDev:  ' + str(np.std(scores)))
            print()
        elif args.tune:
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [100, 10, 1.0, 0.1, 0.01]
            grid = {'solver':solvers,'penalty':penalty,'C':c_values}

            lr = sklm.LogisticRegression(max_iter=10000)
            lr_grid = skms.GridSearchCV(
                estimator=lr, param_grid=grid, n_jobs=-1, cv=3, scoring='accuracy', error_score=0)

            lrm_pred = lrm.predict(test_X)
            print('Base Logistic Regression: ' +
                  str(skm.accuracy_score(test_y, lrm_pred)))

            lr_grid.fit(train_X, train_y)
            # lr_grid_pred = lr_grid.predict(test_X)
            print('Tuned Logistic Regression: ' +
                  str(lr_grid.best_score_))
            print('Tuned Hyperparameters: ' + str(lr_grid.best_params_))
            
        else:
            lrm_pred = lrm.predict(test_X)
            print('Logistic Regression: ' + str(skm.accuracy_score(test_y, lrm_pred)))

    # Random Forest
    if 'rf' in args.models:
        rfc = ske.RandomForestClassifier()
        # rfc = ske.RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=1, max_depth=80, bootstrap=False)
        # {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 80, 'bootstrap': False}
        # {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 80, 'bootstrap': False}
        rfc = rfc.fit(train_X, train_y)
        if args.cross:
            scores = skms.cross_val_score(rfc, train_X, train_y, cv=3)
            print('Random Forest')
            print('Scores: ' + ', '.join([str(score)[:8] for score in scores]))
            print('Mean:   ' + str(np.average(scores)))
            print('StDev:  ' + str(np.std(scores)))
            print()
        elif args.tune:
            n_estimators = [int(x) for x in np.linspace(
                start=200, stop=2000, num=10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}

            rfc_pred = rfc.predict(test_X)
            print('Base Random Forest: ' + str(skm.accuracy_score(test_y, rfc_pred)))
            print('Base Hyperparameters: ' + str())

            rf = ske.RandomForestClassifier()
            rf_random = skms.RandomizedSearchCV(estimator=rf,
                                                param_distributions=random_grid, n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)
            rf_random.fit(train_X, train_y)
            # rf_random_pred = rf_random.predict(test_X)
            print('Tuned Random Forest: ' + str(rf_random.best_score_))
            print('Tuned Hyperparameters: ' + str(rf_random.best_params_))
        else:
            rfc_pred = rfc.predict(test_X)
            print('Random Forest: ' + str(skm.accuracy_score(test_y, rfc_pred)))

    print('Done.')
    return

if __name__ == "__main__":
    main()