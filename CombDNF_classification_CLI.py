### Description: CLI for training and evaluating classification models on drug combination data.
### Authors: Pauline Hiort, TODO: add other authors
### Date: 2024-2025

import CombDNF_utils
import hyperparam_grids as hp
import pandas as pd
import numpy as np
from sklearn.metrics import get_scorer_names
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import sys, click, logging, os
from time import time
from joblib import dump


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '-f',
    '--features_file',
    type=click.Path(exists=True),
    required=True,
    help='Path of the tab separated input file with features.')
@click.option(
    '-g',
    '--ground_truth_file',
    type=click.Path(exists=True),
    required=True,
    help='Path of the tab separated input file.')
@click.option(
    '-o',
    '--output_path',
    type=click.Path(exists=False),
    default='.',
    help='Path to the folder in which the output files should be placed. Default: current directory')
@click.option(
    '-p',
    '--features_only_prediction',
    is_flag=True,
    help='Predict features data without ground truth if given in features_file. Default: False')
@click.option(
    '-pd',
    '--predictions_drugs',
    type=click.Choice(['both', 'one', 'all']),
    default='both',
    help='Drugs combinations to consider for prediction. Predict only drug combinations with "both" or "one" drug(s) given in ground truth data '
    'or predict "all" given drug combinations. Default: both')
@click.option(
    '-pl',
    '--positive_label',
    type=int,
    default=1,
    help='Label for positive class i.e., approved drug combinations. Default: 1')
@click.option(
    '-c',
    '--classification_models',
    type=click.Choice(['all', 'knn', 'lda', 'logreg', 'nb',  'svm', 'rf', 'none']),
    multiple=True,
    default=None,
    help='Additional classification models to be used for prediction. Multiple models can be selected with, e.g., -cm knn -cm svm.' 
    'List of models: k-Neirest Neighbors (knn), Linear Discriminant Analysis (lda), Logistic Regression (logreg), Naive Bayes (nb), Random Forest (rf), Support Vector Machine (svm).'
    'Default: xgb')
@click.option(
    '-sc',
    '--feature_scaling',
    type=click.Choice(['standard', 'none']), 
    default=None,
    help='Scaling method for scaling the data before training the model. Default: None')
@click.option(
    '-ba',
    '--balancing_method',
    type=click.Choice(['adasyn', 'smote', 'smoteenn', 'smotetomek', 'none']),
    default=None,
    help='Balancing method to be used for unbalanced data. Default: None')
@click.option(
    '-se',
    '--feature_selection',
    type=click.Choice(['kbest', 'rfe', 'none']),
    default=None,
    help='Feature selection method to be used for model optimization. Default: None')
@click.option(
    '-t',
    '--tuning_method',
    type=click.Choice(['bayes', 'grid', 'none']),
    default='grid',
    help='Hyperparameter tuning method to be used for model optimization. Default: grid')
@click.option(
    '-sm',
    '--scoring_method',
    type=click.Choice(get_scorer_names()),
    default='matthews_corrcoef',
    help='Scoring method to be used for model optimization and evaluation. Default: matthews_corrcoef')
@click.option(
    '-ns',
    '--n_splits',
    type=int,
    default=5,
    help='Number of splits for cross-validation. Default: 5')
@click.option(
    '-nv',
    '--nested_cross_validation',
    is_flag=True,
    help='Use nested cross-validation for model evaluation. Default: False')
@click.option(
    '-r',
    '--random_seed',
    type=int,
    default=123,
    help='Set random seed for reproducibility. Default: 123')
@click.option(
    '-cpu', 
    '--n_cpus',
    type=int,
    default=1,
    help='Number of Processes to use for parallel processing. Default: 1')


# function for one model and/or separate feature data
def main(
    features_file: str, 
    ground_truth_file: str, 
    output_path: str, 
    features_only_prediction: bool,
    predictions_drugs: str,
    positive_label: str,
    classification_models: tuple, 
    feature_scaling: str, 
    balancing_method: str, 
    feature_selection: str,
    tuning_method: str, 
    scoring_method: str,
    n_splits: int,
    nested_cross_validation: bool,
    random_seed: int,
    n_cpus: int):
    """
    Commandline tool for training and evaluating classification models on drug combination data.
    Example usage: python CombDNF_classification_CLI.py -f features.tsv -g ground_truth.tsv -o output_folder -p -pd both -pl 1 -c all -ba adasyn -t grid -sm matthews_corrcoef -ns 5 -r 123 -cpu 1

    input:
        features_file (str) - path of the tab separated input file with features
        ground_truth_file (str) - path of the tab separated input file
        output_path (str) - path to the folder in which the output files should be placed
        features_only_prediction (bool) - predict features data without ground truth if given in features_file
        predictions_drugs (str) - drugs combinations to consider for prediction
        positive_label (str) - label for positive class i.e., approved drug combinations
        classification_models (tuple) - classification model to be used for prediction
        feature_scaling (str) - scaling method for scaling the data before training the model
        balancing_method (str) - balancing method to be used for unbalanced data
        feature_selection (str) - feature selection method to be used for model optimization
        tuning_method (str) - hyperparameter tuning method to be used for model optimization
        scoring_method (str) - scoring method to be used for model optimization and evaluation
        n_splits (int) - number of splits for cross-validation
        nested_cross_validation (bool) - use nested cross-validation for model evaluation
        random_seed (int) - set random seed for reproducibility
        n_cpus (int) - number of Processes to use for parallel processing
    output:
        None
    """

    ### start time for runtime logging
    start_t = time()
    
    ### create output folder if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        # logging.info(f"Created output folder {output_path}.")
    

    ### set up logging to command line and file
    logging.basicConfig(
        handlers=[logging.FileHandler(f'{output_path}/logging.log'),
                  logging.StreamHandler()], 
        level=logging.INFO, 
        format='[%(asctime)s] %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info(f"Starting CombDNF classification training and evaluation ...")

    
    ### check if all classification models should be used
    classification_models = None if classification_models=='none' else classification_models
    if classification_models is not None:
        classification_models = list(classification_models)
        if 'all' in classification_models:
            classification_models = ['knn', 'lda', 'logreg', 'nb', 'rf', 'svm']
        classification_models = ['xgb'] + classification_models
    else:
        classification_models = ['xgb']

    logging.info(f"Output files will be saved in {output_path}")
    logging.info(f"The log is saved in {output_path}/logging.log")
    logging.info(f"Configurations: \n"
                 f"Features file: {features_file}\n"
                 f"Ground truth file: {ground_truth_file}\n"
                 f"Features only prediction: {features_only_prediction}\n"
                 f"Predictions for drug combinations: {predictions_drugs}\n"
                 f"Positive label: {positive_label}\n"
                 f"Classification models: {classification_models}\n"
                 f"Feature scaling: {feature_scaling}\n"
                 f"Balancing method: {balancing_method}\n"
                 f"Feature selection: {feature_selection}\n"
                 f"Tuning method: {tuning_method}\n"
                 f"Scoring method: {scoring_method}\n"
                 f"Number of cross-validation splits: {n_splits}\n"
                 f"Nested cross-validation: {nested_cross_validation}\n"
                 f"Random seed: {random_seed}\n"
                 f"Number of CPUs/Processes: {n_cpus}")
    
    ### process feature data and ground truth data
    logging.info(f"Importing and merging feature data and ground truth data ...")
    merged_df, features_only_df = CombDNF_utils.merge_features_groundtruth(features_file, ground_truth_file, features_only_prediction, predictions_drugs)
    logging.info(f"Importing and merging feature data and ground truth data ... successful in {(time() - start_t)/60:.2f} minutes.")
    
    ### split feature and ground truth data for training and testing
    X, y = CombDNF_utils.split_X_y(merged_df)
    logging.info(f"Ground truth label counts: {y.value_counts().to_dict()}")
    
    ### swap drugA and drugB labels for the data
    logging.info(f"Swapping drugA and drugB labels and all corresponding features ...")
    X, y = CombDNF_utils.swap_labels(X, y)
    logging.info(f"Shape of merged and swapped data: {X.shape}")
    logging.info(f"Swapping drugA and drugB labels and all corresponding features ... successful.")
    merged_swapped_df = pd.concat([X, y], axis=1).sort_index(level='drugcomb_sorted', axis=0)
    merged_swapped_df['original_drugcombo'] = False
    merged_swapped_df.loc[merged_df.index, 'original_drugcombo'] = True
    merged_swapped_df = merged_swapped_df.loc[:, ['label', 'original_drugcombo']]
    merged_swapped_df.to_csv(f"{output_path}/predictions_test_data.csv", sep='\t', index=True)
    
    ### merge feature and ground truth data for predictions
    if features_only_prediction:
        predictions_df = features_only_df
        X_pred_only = predictions_df.copy()
    del features_only_df

    ### split data into n_splits stratified groups for cross-validation
    cv_outer = StratifiedGroupKFold(n_splits=n_splits)
    # X = X.sample(frac=1, random_state=random_seed)
    # y = y.sample(frac=1, random_state=random_seed)
    all_folds_idx = []
    for _, test_ix in cv_outer.split(X, y, groups=X.index.get_level_values('drugcomb_sorted')):
        all_folds_idx.append(test_ix)
    
    ### create list for pipeline
    pipe_list = []

    ### add feature scaling, feature selection and balancing method to pipeline
    feature_selection = None if feature_selection == 'none' else feature_selection
    feature_scaling = None if feature_scaling == 'none' else feature_scaling
    balancing_method = None if balancing_method == 'none' else balancing_method
    
    if feature_scaling is not None:
        pipe_list.append(CombDNF_utils.return_scaler(feature_scaling))

    if balancing_method is not None:
        pipe_list.append(CombDNF_utils.return_balancer(balancing_method, random_seed))

    if feature_selection is not None:
        n_features = X.shape[1]
        pipe_fselection, hyperparam_fselection = CombDNF_utils.return_fselecter(feature_selection, n_features, random_seed)
        pipe_list.append(pipe_fselection)
        logging.info(f"Feature selection: {pipe_fselection} to {n_features} features.")
        logging.info(f"Hyperparameters for feature selection: {hyperparam_fselection}.")

    ### create empty dataframe for scores
    test_scores = pd.DataFrame({'model': [], 'fold': []})

    labels = sorted(list(set(y)))

    for cl_model in classification_models:
        
        logging.info(f"Training and testing classification model {cl_model} ...")
        t1 = time()

        ### add classification model to pipeline
        if cl_model == 'knn':
            pipe_list.append(('classify_knn', KNeighborsClassifier()))

            hyperparameters = hp.HYPERPARAMETERS_KNN
            hyperparameters_bayes = hyperparameters

        elif cl_model == 'lda':
            pipe_list.append(('classify_lda', LinearDiscriminantAnalysis()))
        
            hyperparameters = hp.HYPERPARAMETERS_LDA
            try: hyperparameters_bayes = [(hyperparameters[0], 1), (hyperparameters[1], 15)]
            except: hyperparameters_bayes = hyperparameters

        elif cl_model == 'logreg':
            pipe_list.append(('classify_logreg', LogisticRegression(random_state=random_seed))) #TODO: class_weight='balanced'??

            hyperparameters = hp.HYPERPARAMETERS_LOGISTIC_REGRESSION
            try: hyperparameters_bayes = [(hyperparameters[0], 10), (hyperparameters[1], 20)]
            except: hyperparameters_bayes = hyperparameters
            
        elif cl_model == 'nb':
            pipe_list.append(('classify_nb', GaussianNB()))

            hyperparameters = hp.HYPERPARAMETERS_NAIVE_BAYES
            try: hyperparameters_bayes = [(hyperparameters[0], 10)]
            except: hyperparameters_bayes = hyperparameters

        elif cl_model == 'rf':
            pipe_list.append(('classify_rf', RandomForestClassifier(random_state=random_seed)))

            hyperparameters = hp.HYPERPARAMETERS_RANDOM_FOREST
            hyperparameters_bayes = hyperparameters #[(hyperparameters[0], 5)]

        elif cl_model == 'svm':
            pipe_list.append(('classify_svm', SVC(random_state=random_seed)))

            hyperparameters = hp.HYPERPARAMETERS_SVM
            hyperparameters_bayes = hyperparameters

        elif cl_model == 'xgb':
            pipe_list.append(('classify_xgb', XGBClassifier(random_state=random_seed)))

            hyperparameters = hp.HYPERPARAMETERS_XGBOOST
            hyperparameters_bayes = hyperparameters

        if feature_selection is not None:
            ### update hyperparameters with selected features
            hyperparameters = [{**hyperparam_fselection, **d} for d in hyperparameters]
            hyperparameters_bayes = [{**hyperparam_fselection, **d} if not isinstance(d, tuple) else ({**hyperparam_fselection, **d[0]}, d[1]) for d in hyperparameters_bayes]
        
        pipe = Pipeline(pipe_list)
        
        logging.info(f"Cross-validation with hyperparameter tuning and testing for {cl_model} ...")
        
        model_val_scores = pd.DataFrame({'fold': [], 
                                         'best_model': [], 
                                         f'valset_mean_{scoring_method}_score': [], 
                                         f'valset_std_{scoring_method}_score': [], 
                                         'rank': [], 
                                         'params': []})

        model_test_scores = []
        
        ### figure for ROC and PR curves
        fig, ax = plt.subplots(2, 2, sharey=False, figsize=(12, 12))

        ### loop over all k-folds
        for i in range(len(all_folds_idx)): #[2:]
            t2 = time()
            logging.info(f"Cross-validation fold {i+1}/{n_splits} ...")
            ### get train, validation and test indices from k-fold test splits
            train_val_idx = np.concatenate(all_folds_idx[:i]+all_folds_idx[i+1:])
            ### if last fold, use first fold as validation set
            if i+1 == len(all_folds_idx): val_idx = all_folds_idx[0]
            else: val_idx = all_folds_idx[i+1]
            test_idx = all_folds_idx[i]

            ### for test set remove one random drug combination of swapped drug labels
            test_idx_removed_swapped = np.array(y.reset_index(drop=False).iloc[test_idx].groupby('drugcomb_sorted').sample(1, random_state=random_seed).index)

            ### inner cross-validation fold for hyperparameter tuning
            ### get train indices from entire dataset and map to index in train set only
            train_set_idx_map = y.reset_index(drop=True).iloc[train_val_idx].reset_index(drop=False).drop(columns='label')
            inner_val_idx = np.array(train_set_idx_map[train_set_idx_map['index'].isin(val_idx)].index)
            inner_train_idx = np.array(train_set_idx_map[~train_set_idx_map['index'].isin(val_idx)].index)

            ### train-test split for outer cross-validation
            X_train, y_train = X.iloc[train_val_idx], y.iloc[train_val_idx]
            X_test, y_test = X.iloc[test_idx_removed_swapped], y.iloc[test_idx_removed_swapped]
            
            if tuning_method == 'grid':
                if not nested_cross_validation:
                    ### hyperparameter tuning with grid search on train-validation set
                    grid = GridSearchCV(pipe,
                                        param_grid=hyperparameters, 
                                        scoring=scoring_method,
                                        n_jobs=n_cpus, 
                                        refit=True,
                                        cv=[(inner_train_idx, inner_val_idx)],
                                        verbose=0)
                else:
                    ### hyperparameter tuning with grid search with cross-validation on train set
                    grid = GridSearchCV(pipe,
                                        param_grid=hyperparameters, 
                                        scoring=scoring_method,
                                        n_jobs=n_cpus, 
                                        refit=True,
                                        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, 
                                                                random_state=random_seed), 
                                        verbose=0)
            elif tuning_method == 'bayes':
                if not nested_cross_validation:
                    ### hyperparameter tuning with bayes search on train-validation set
                    grid = BayesSearchCV(pipe,
                                        search_spaces=hyperparameters_bayes,
                                        n_iter=25,
                                        scoring=scoring_method, 
                                        n_jobs=n_cpus, 
                                        refit=True, 
                                        cv=[(inner_train_idx, inner_val_idx)], 
                                        verbose=0, 
                                        random_state=random_seed)
                else:
                    ### hyperparameter tuning with bayes search with cross-validation on train set
                    grid = BayesSearchCV(pipe,
                                        search_spaces=hyperparameters_bayes,
                                        n_iter=50,
                                        scoring=scoring_method, 
                                        n_jobs=n_cpus, 
                                        refit=True, 
                                        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, 
                                                                random_state=random_seed), 
                                        verbose=0, 
                                        random_state=random_seed)
            else:
                hyperparameters = [{}]
                grid = GridSearchCV(pipe,
                                    param_grid=hyperparameters, 
                                    scoring=scoring_method,
                                    n_jobs=n_cpus, 
                                    refit=True,
                                    cv=[(inner_train_idx, inner_val_idx)],
                                    verbose=0)
        
            ### fit model
            grid.fit(X_train, y_train, groups=X_train.index.get_level_values('drugcomb_sorted'))

            ### get best hyperparameters and scores and save in dataframe
            best_params = ['*' if prams == grid.best_params_ else np.nan for prams in grid.cv_results_['params']]
            model_val_scores = pd.concat([model_val_scores, 
                                          pd.DataFrame({
                                              'fold': int(i), 
                                              'best_model': best_params,
                                              f'valset_mean_{scoring_method}_score': np.round(grid.cv_results_['mean_test_score'], 5), 
                                              f'valset_std_{scoring_method}_score': np.round(grid.cv_results_['std_test_score'], 5), 
                                              'rank': grid.cv_results_['rank_test_score'], 
                                              'params': grid.cv_results_['params']})], 
                                          axis=0, ignore_index=True)

            if feature_selection is not None:
                selected_features = grid.best_estimator_.named_steps['select'].get_feature_names_out()
                logging.info(f"Selected {len(selected_features)} features for fold {i+1}/{n_splits}: {selected_features}")
            
            ### get best model and predict test set
            y_pred = grid.predict(X_test)
            if not cl_model == 'svm':
                y_pred_prob = grid.predict_proba(X_test)
            else:
                y_pred_prob = None

            # logging.info(f"Prediction on test set without swapped drug labels for fold {i+1}/{n_splits} ... successful.")

            ### calculate scores for test set without swapped drug labels
            model_test_scores.append(grid.score(X_test, y_test))
            accuracy, accuracy_balanced, f1_binary, f1_weighted, mcc, auroc_cl, auprc_cl, auroc_pb, auprc_pb, ax = CombDNF_utils.calculate_scores_prob(y_test, y_pred, y_pred_prob, positive_label=positive_label, output_path=output_path, model=cl_model, fold=i, fig=fig, ax=ax)

            test_scores = pd.concat([test_scores,
                                     pd.DataFrame({
                                         'model': cl_model,
                                         'fold': int(i),
                                         'accuracy': accuracy, 
                                         'accuracy_balanced': accuracy_balanced,
                                         'f1_binary': f1_binary,
                                         'f1_weighted': f1_weighted, 
                                         'matthews_corrcoef': mcc, 
                                         'auroc_class': auroc_cl, 
                                         'auprc_class': auprc_cl, 
                                         'auroc_prob': auroc_pb,
                                         'auprc_prob': auprc_pb}, index=[i])], 
                                     axis=0, ignore_index=True)
            
            ### predict test set with swapped drug labels for output
            merged_swapped_df.loc[X.iloc[test_idx_removed_swapped].index, 'test_set_in_fold'] = i
            merged_swapped_df.loc[X.iloc[test_idx].index, f'prediction_{cl_model}'] = grid.predict(X.iloc[test_idx])
            if not cl_model == 'svm':
                try: merged_swapped_df.loc[X.iloc[test_idx].index, [f'prediction_prob_label{elem:.0f}_{cl_model}' for elem in labels]] = np.round(grid.predict_proba(X.iloc[test_idx]), 6)
                except: merged_swapped_df.loc[X.iloc[test_idx].index, [f'prediction_prob_label{elem}_{cl_model}' for elem in labels]] = np.round(grid.predict_proba(X.iloc[test_idx]), 6)
            else:
                logging.info(f"No probability prediction for SVM model due to inconsistency in sklearn's implementation.")
            logging.info(f"Cross-validation fold {i+1}/{n_splits} ... successful in {(time() - t2)/60:.2f} minutes.")

            ### predict all features data for output
            if features_only_prediction:
                t2 = time()
                predictions_df.loc[X_pred_only.index, f'prediction_{cl_model}_fold{i}'] = grid.predict(X_pred_only)
                
                if not cl_model == 'svm':
                    try:
                        predictions_df.loc[X_pred_only.index, [f'prediction_prob_label{elem:.0f}_{cl_model}_fold{i}' for elem in labels]] = np.round(grid.predict_proba(X_pred_only), 6)
                    except:
                        predictions_df.loc[X_pred_only.index, [f'prediction_prob_label{elem}_{cl_model}_fold{i}' for elem in labels]] = np.round(grid.predict_proba(X_pred_only), 6)
                logging.info(f"Prediction of all feature data for fold {i+1}/{n_splits} ... successful in {(time() - t2)/60:.2f} minutes.")
                
            ### save best model for fold
            dump(grid.best_estimator_, f"{output_path}/best_model_{cl_model}_fold{i}.joblib")
            logging.info(f"Best model for fold {i+1}/{n_splits} saved in 'best_model_{cl_model}_fold{i}.joblib'.")

        logging.info(f"Mean {scoring_method} test score for {cl_model}: {np.mean(model_test_scores):.3f} ({np.std(model_test_scores):.3f})")
        
        ### save scores and parameters for all tested hyperparameters
        model_val_scores.to_csv(f"{output_path}/predictions_tuning_scores_{cl_model}.csv", sep='\t', index=False)
        ### save test scores for all folds
        test_scores.to_csv(f"{output_path}/predictions_testset_scores.csv", sep='\t', index=False)
        ### save predictions for all input features and models
        merged_swapped_df.to_csv(f"{output_path}/predictions_test_data.csv", sep='\t', index=True)
        ### save predictions for features only data
        predictions_df.to_csv(f"{output_path}/predictions_all_data.csv", sep='\t', index=True)

        logging.info(f"Scores and hyperparameters for method {cl_model} saved in 'predictions_tuning_scores_{cl_model}.csv'.")
        logging.info(f"Cross-validation with hyperparameter tuning and testing for {cl_model} ... successful.")
        
        model_test_scores = test_scores[test_scores['model'] == cl_model]
        metrics_text = (f"Mean scores (std) for hold-out test predictions of all folds: \n"
                        f"Accuracy: {np.mean(model_test_scores.accuracy):.3f} ({np.std(model_test_scores.accuracy):.3f})\n"
                        f"Balanced Accuracy: {np.mean(model_test_scores.accuracy_balanced):.3f} ({np.std(model_test_scores.accuracy_balanced):.3f})\n"
                        f"F1-score (binary): {np.mean(model_test_scores.f1_binary):.3f} ({np.std(model_test_scores.f1_binary):.3f})\n"
                        f"F1-score (weighted): {np.mean(model_test_scores.f1_weighted):.3f} ({np.std(model_test_scores.f1_weighted):.3f})\n"
                        f"Matthews Correlation Coefficient: {np.mean(model_test_scores.matthews_corrcoef):.3f} ({np.std(model_test_scores.matthews_corrcoef):.3f})\n"
                        f"AUROC (class): {np.mean(model_test_scores.auroc_class):.3f} ({np.std(model_test_scores.auroc_class):.3f})\n"
                        f"AUPRC (class): {np.mean(model_test_scores.auprc_class):.3f} ({np.std(model_test_scores.auprc_class):.3f})\n"
                        f"AUROC (prob): {np.mean(model_test_scores.auroc_prob):.3f} ({np.std(model_test_scores.auroc_prob):.3f})\n"
                        f"AUPRC (prob): {np.mean(model_test_scores.auprc_prob):.3f} ({np.std(model_test_scores.auprc_prob):.3f})")
        logging.info(metrics_text)

        fig.savefig(f"{output_path}/roc_pr_curve_{cl_model}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        ### remove classification model from pipeline for next model
        pipe_list = pipe_list[:-1]
        
        logging.info(f"Training and testing classification model {cl_model} ... successful in {(time() - t1)/60:.2f} minutes..")
        # run_t = (time() - start_t)/60
        # logging.info(f"Runtime for training and validating {cl_model}: {run_t:.2f} min.")

    ### save predictions for all input features and models
    # merged_swapped_df.to_csv(f"{output_path}/predictions_test_data.csv", sep='\t', index=True)
    logging.info(f"Predictions for all input features and models saved in 'predictions_test_data.csv'.")

    ### save scores for all models
    # test_scores.to_csv(f"{output_path}/predictions_testset_scores.csv", sep='\t', index=False)
    logging.info(f"Test set scores for all models and cross-validation folds saved in 'predictions_testset_scores.csv'.")

    ### save predictions for features only data
    # predictions_df.to_csv(f"{output_path}/predictions_all_data.csv", sep='\t', index=True)
    logging.info(f"Predictions for all features data for all models saved in 'predictions_all_data.csv'.")

    logging.info(f"Classification model training and evaluation ... successful in {(time() - start_t)/60:.2f} minutes.\n")


# function to ensure, script is only executable over commandline
if __name__ == "__main__":
    main()
