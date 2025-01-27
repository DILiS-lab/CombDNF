import numpy as np

HYPERPARAMETERS_KNN = [{'classify_knn__n_neighbors': list(np.arange(2, 30, 3)),
                        'classify_knn__weights': ['uniform', 'distance'],
                        'classify_knn__p': [1, 2],
                        'classify_knn__metric': ['minkowski']},
                       {'classify_knn__n_neighbors': list(np.arange(2, 30, 3)),
                        'classify_knn__weights': ['uniform', 'distance'],
                        'classify_knn__metric': ['cosine']}]

HYPERPARAMETERS_LDA = [{'classify_lda__solver': ['svd'], },
                       {'classify_lda__solver': ['lsqr', 'eigen'], 
                        'classify_lda__shrinkage': [None, 'auto'] + list(np.arange(0, 1, 0.1))}]

HYPERPARAMETERS_LOGISTIC_REGRESSION = [{'classify_logreg__C': [100, 10, 1, 0.1, 0.01], 
                           'classify_logreg__penalty': ['l1', 'l2'],
                           'classify_logreg__solver': ['liblinear', 'saga'],
                           'classify_logreg__max_iter': [100000], 
                           'classify_logreg__class_weight': ['balanced', None]},
                          {'classify_logreg__C': [100, 10, 1, 0.1, 0.01],
                           'classify_logreg__penalty': ['l2'],
                           'classify_logreg__solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
                           'classify_logreg__max_iter': [100000], 
                           'classify_logreg__class_weight': ['balanced', None]}]
                                                        
HYPERPARAMETERS_NAIVE_BAYES = [{'classify_nb__var_smoothing': list(np.logspace(-9, 1, 10))}]

HYPERPARAMETERS_SVM = [{'classify_svm__C': [100, 10, 1, 0.1, 0.01],
                        'classify_svm__gamma': ['scale', 10, 1, 0.1, 0.01],
                        'classify_svm__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                        'classify_svm__max_iter': [100000], 
                        'classify_svm__class_weight': ['balanced', None]}]

HYPERPARAMETERS_RANDOM_FOREST = [{'classify_rf__n_estimators': [10, 100, 500],  
                                  'classify_rf__criterion': ['gini', 'entropy'],
                                  'classify_rf__max_features': ['sqrt', None],
                                  'classify_rf__max_depth': [5, 10, 50],
                                  'classify_rf__min_samples_split': [2, 10],
                                  'classify_rf__min_samples_leaf': [1, 3],
                                  'classify_rf__bootstrap': [True, False], 
                                  'classify_rf__class_weight': ['balanced', None]}]

HYPERPARAMETERS_XGBOOST = [{'classify_xgb__learning_rate': [0.001, 0.01, 0.1],
                            'classify_xgb__n_estimators': [10, 100, 500],
                            'classify_xgb__max_depth': [5, 10], 
                            'classify_xgb__min_child_weight': [5, 10], 
                            'classify_xgb__gamma': [0.0, 0.2, 0.4],
                            'classify_xgb__subsample': [0.5, 0.7, 0.9],
                            'classify_xgb__colsample_bytree': [0.5, 0.7, 0.9],
                            'classify_xgb__reg_alpha': [0.0, 0.5, 1.0],
                            'classify_xgb__reg_lambda': [0.0, 0.5, 1.0]
                           }]