### Description: CLI for training and evaluating classification models on drug combination data.
### Authors: Pauline Hiort, TODO: add other authors
### Date: 2024-2025

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import logging, os


def create_drug_comb_sorted_column(df: pd.DataFrame):
    """
    create new column 'drugcomb_sorted' with sorted drug names
    input:
        df (pandas dataframe) - dataframe with drug combination data
    output:
        df (pandas dataframe) - dataframe with new column 'drugcomb_sorted'
    """
    df['drugcomb_sorted'] = df.apply(lambda x: '_'.join(sorted([x['drugA'], x['drugB']])), axis=1)
    return df

def merge_features_groundtruth(
        feature_file_path: os.PathLike, 
        groundtruth_file_path: os.PathLike, 
        features_only_prediction: bool = False, 
        predictions_drugs: str = 'both'):
    """
    merge feature and ground truth data based on the 'drugcomb_sorted' column
    input:
        feature_file_path (os.PathLike) - path to the feature file
        groundtruth_file_path (os.PathLike) - path to the ground truth file
        features_only_prediction (bool) - boolean value indicating whether only feature data should be used for prediction
        predictions_drugs (str) - string indicating which drugs should be used for prediction ('both', 'one', 'all')
    output:
        merged_data (pandas dataframe) - dataframe with merged feature and ground truth data
        features_only_data (pandas dataframe) - dataframe with feature
    """
    ### load feature and ground truth data
    features = pd.read_csv(feature_file_path, sep='\t')
    groundtruth = pd.read_csv(groundtruth_file_path,sep='\t')
    logging.info(f"Loaded feature data with shape {features.shape} and ground truth data with shape {groundtruth.shape}.")
    
    ### check if feature and ground truth data contain necessary columns
    if 'drugA' not in features.columns or 'drugB' not in features.columns:
        logging.info(f"ERROR: Feature data must contain columns 'drugA' and 'drugB'!")
        raise ValueError("Feature data must contain columns 'drugA' and 'drugB'!")
    if 'drugA' not in groundtruth.columns or 'drugB' not in groundtruth.columns or 'label' not in groundtruth.columns:
        logging.info(f"ERROR: Ground truth data must contain columns 'drugA', 'drugB' and 'label'!")
        raise ValueError("Ground truth data must contain columns 'drugA', 'drugB' and 'label'!")

    ### create 'drugcomb_sorted' columns for feature data
    feature_data = create_drug_comb_sorted_column(features)
    
    ### create 'drugcomb_sorted' columns for ground truth data
    ground_truth = create_drug_comb_sorted_column(groundtruth)
    ground_truth = ground_truth.drop_duplicates(subset=['drugcomb_sorted', 'label'])
    ambiguous = ground_truth[ground_truth.duplicated(subset=['drugcomb_sorted'], keep=False)].index
    if len(ambiguous) > 0:
        ground_truth = ground_truth.drop(ambiguous)
        logging.info(f"Removed {len(ambiguous)} ambiguous entries from ground truth data.")
    ground_truth = ground_truth[['drugcomb_sorted', 'label']]
    
    ### merge feature and ground truth data
    merged_data_all = feature_data.merge(ground_truth, on='drugcomb_sorted', how="left")
    merged_data = merged_data_all.dropna(subset=['label'])
    if features_only_prediction:
        features_only_data = merged_data_all[merged_data_all['label'].isna()].drop('label', axis=1)
        gt_drugs = list(set(merged_data.drugA.to_list() + merged_data.drugB.to_list()))
        if predictions_drugs == 'both':
            features_only_data = features_only_data[features_only_data['drugA'].isin(gt_drugs) & features_only_data['drugB'].isin(gt_drugs)]
        elif predictions_drugs == 'one':
            features_only_data = features_only_data[features_only_data['drugA'].isin(gt_drugs) | features_only_data['drugB'].isin(gt_drugs)]
        elif predictions_drugs == 'all':
            features_only_data = features_only_data
        ### check if drugs in feature data are in ground truth data
        if len(set(features_only_data.drugA.to_list() + features_only_data.drugB.to_list()) - set(gt_drugs)) > 0:
            logging.info(f"CAUTION: Predictions of new drug combinations will include some drugs that are not in ground truth data!")

        ### check if there are any drug combinations left for prediction
        if features_only_data.shape[0] == 0:
            logging.info(f"ERROR: No new drug combinations left for prediction after filtering for drug combinations in ground truth data!")
            logging.info(f"ERROR: No new drug combinations will be predicted!")
            features_only_data = None
    else:
        features_only_data = None
    
    ### clean merged datasets for NaN values
    merged_data = merged_data.dropna(axis=0, how='all').dropna(axis=1, how='all')
    logging.info(f"Shape of merged feature and ground truth data: {merged_data.shape}")
    ### set index for merged data
    merged_data.set_index(['drugA', 'drugB', 'drugcomb_sorted'], inplace=True)
    
    ### clean feature only data for NaN values
    if features_only_prediction and features_only_data is not None:        
        features_only_data = features_only_data.dropna(axis=0, how='all').dropna(axis=1, how='all')
        logging.info(f"Shape of feature only data: {features_only_data.shape}")
        ### set index for feature only data
        features_only_data.set_index(['drugA', 'drugB', 'drugcomb_sorted'], inplace=True)
    
    return merged_data, features_only_data


def split_X_y(df: pd.DataFrame):
    """
    split the merged data into feature matrix X and target vector Y
    input:
        df (pandas dataframe) - dataframe with merged feature and ground truth data
    output:
        X (pandas dataframe) - dataframe with feature data
        y (pandas series) - series with target data
    """    
    X = df.drop(['label'], axis=1)
    y = df['label']
    return X, y


def swap_labels(X: pd.DataFrame, y: pd.Series):
    """
    swap labels for drugA/drugB and all corresponding features
    input:
        X (pandas dataframe) - dataframe with feature data
        y (pandas series) - series with target data
    output:
        X_new (pandas dataframe) - dataframe with swapped feature data
        y_new (pandas series) - series with swapped target data
    """    
    ### concatenate feature and target data
    ### reset index for swapping labels of drugA/drugB and all corresponding features
    df = pd.concat([X, y], axis=1)
    df.reset_index(inplace=True)

    ### extract columns for swapping labels of drugA/drugB and all corresponding features
    a_columns = [col for col in df.columns if 'A' in col and 'B' not in col]
    b_columns = [col for col in df.columns if 'B' in col and 'A' not in col]

    ### create mapping for swapping columns
    swap_mapping = {**{col: col.replace('A', 'B') for col in a_columns}, 
                    **{col: col.replace('B', 'A') for col in b_columns}}
    
    ### apply mapping to swap columns and reset index for concatenatation
    swapped_df = df.rename(columns=swap_mapping)
    swapped_df.set_index(['drugA', 'drugB', 'drugcomb_sorted'], inplace=True)
    df.set_index(['drugA', 'drugB', 'drugcomb_sorted'], inplace=True)

    ### concatenate the original and swapped dataframes and split into X and y
    augmented_df = pd.concat([df, swapped_df], ignore_index=False)
    
    X_new, y_new = split_X_y(augmented_df)
    
    return X_new, y_new


def return_scaler(method: str):
    """
    retrun scaler for sklearn pipeline
    input:
        method (string) - string indicating the scaling method
    output:
        tuple (tuple) - tuple with scaling method and object
    """
    if method == 'standard':
        return ('scale', StandardScaler())
    else:
        return None


def return_balancer(method: str, random_seed: int):
    """
    return balancer for sklearn pipeline
    input:
        method (string) - string indicating the balancing method
        random_seed (int) - integer value for random seed
    output:
        tuple (tuple) - tuple with balancing method and object
    """
    if method == 'adasyn':
        return ('balance', ADASYN(random_state=random_seed))
    elif method == 'smote':
        return ('balance', SMOTE(random_state=random_seed))
    elif method == 'smoteenn':
        return ('balance', SMOTEENN(random_state=random_seed))
    elif method == 'smotetomek':
        return ('balance', SMOTETomek(random_state=random_seed))


def return_fselecter(method: str, n_features: int, random_seed: int):
    """
    return feature selection method for sklearn pipeline
    input:
        method (string) - string indicating the feature selection method
        n_features (int) - integer value for number of features
        random_seed (int) - integer value for random seed
    output:
        tuple (tuple), dict (dict) - tuple with feature selection method and obeject and dictionary with parameters
    """

    if method == 'kbest':
        return ('select', SelectKBest(score_func=mutual_info_classif)), {'select__k': [int(round(n_features*i)) for i in np.arange(0.2, 0.81, 0.1)]}
    elif method == 'rfe':
        return ('select', RFE(estimator=DecisionTreeClassifier(random_state=random_seed), step=1)), {'select__n_features_to_select': [int(round(n_features*i))  for i in np.arange(0.2, 0.81, 0.1)]}
    else:
        return None


def calculate_scores_prob(
        y_test: pd.Series, 
        y_pred: pd.Series,  
        y_pred_proba: pd.Series = None, 
        positive_label: int = 1, 
        output_path: str = '.', 
        model: str = '', 
        fold: str = '', 
        fig: plt.figure = None, 
        ax: plt.axis = None):
    """
    calculate accuracy, f1_weighted, mcc, auroc, aupr. Create confusion matrix, ROC and PR curve and save as png.
    input:
        y_test (pandas series) - series with target data for testing
        y_pred (pandas series) - series with predicted target data
        y_pred_proba (pandas series) - series with predicted probabilities
        positive_label (int) - integer value for positive label
        output_path (string) - string with output path
        model (string) - string with model name
        fold (string) - string with fold number
        fig (matplotlib figure) - figure object for plotting
        ax (matplotlib axis) - axis object for plotting
    output:
        accuracy (float) - float value for accuracy
        f1_weighted (float) - float value for f1_weighted
        mcc (float) - float value for mcc
        auroc (float) - float value for auroc
        aupr (float) - float value for aupr
    """

    labels = sorted(y_test.unique().tolist())
    pos_idx = labels.index(positive_label)

    ### calculate scores
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_binary = f1_score(y_test, y_pred, average='binary', pos_label=positive_label)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    auroc_class = roc_auc_score(y_test, y_pred)
    auprc_class = average_precision_score(y_test, y_pred)
    if y_pred_proba is not None:
        auroc_proba= roc_auc_score(y_test, y_pred_proba[:, pos_idx])
        auprc_proba = average_precision_score(y_test, y_pred_proba[:, pos_idx])
    else:
        auroc_proba = np.nan
        auprc_proba = np.nan


    if fold == 'all':
        ### create confusion matrix and save as png
        cm_best = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm_best).plot(cmap='Blues', ax=ax1, colorbar=False)
        plt.title(f'Confusion Matrix \nhold-out test data of all folds \nmethod: {model}')
        ax1.text(-0.5, 2.5, f"Scores for model: {model}\nAccuray: {accuracy}\nAccuray (balanced): {accuracy_balanced}\nF1 (pos. label): {f1_binary}\nF1 (weighted): {f1_weighted}\nMCC: {mcc}\nAUROC: {auroc_class}\nAUPR: {auprc_class}")
        plt.savefig(f"{output_path}/confusion_matrix_{model}_all_folds.png", format='png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        #logging.info(f"Confusion matrix saved in {output_path}/confusion_matrix_{model}_fold_{fold}.png")

    if fold == 0:
        ax[0][0].plot([0,1], [0,1], linestyle='--', linewidth=2, color='black', label='random guess', alpha=.8)
    ax[0][0].set_title(f'Receiver operating characteristics (ROC) curve \nclass prediction of hold-out test data per fold')
    RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax[0][0], pos_label=positive_label, label=f"{model} fold {fold} AUROC={auroc_class:.3f}")
    ax[0][0].set_xlabel('False Positive Rate')
    ax[0][0].set_ylabel('True Positive Rate')
    ax[0][0].legend(loc=4)

    if y_pred_proba is not None:
        if fold == 0:
            ax[1][0].plot([0,1], [0,1], linestyle='--', linewidth=2, color='black', label='random guess', alpha=.8)
        ax[1][0].set_title(f'Receiver operating characteristics (ROC) curve \nprobabilty prediction of hold-out test data per fold')
        RocCurveDisplay.from_predictions(y_test, y_pred_proba[:, pos_idx], ax=ax[1][0], pos_label=positive_label, label=f"{model} fold {fold} AUROC={auroc_proba:.3f}")
        ax[1][0].set_xlabel('False Positive Rate')
        ax[1][0].set_ylabel('True Positive Rate')
        ax[1][0].legend(loc=4)
        
    ax[0][1].set_title(f'Precision-recall curve \nclass prediction of hold-out test data per fold')
    PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=ax[0][1], pos_label=positive_label, label=f"{model} fold {fold} AUPRC={auprc_class:.3f}")
    rdg = len(y_test[y_test==positive_label])/len(y_test)
    ax[0][1].set_xlabel('Recall')
    ax[0][1].set_ylabel('Precision')
    ax[0][1].legend(loc=4)

    if y_pred_proba is not None:
        ax[1][1].set_title(f'Precision-recall curve \nprobability prediction of hold-out test data per fold')
        PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba[:, pos_idx], ax=ax[1][1], pos_label=positive_label, label=f"{model} fold {fold} AUPRC={auprc_proba:.3f}")
        ax[1][1].set_xlabel('Recall')
        ax[1][1].set_ylabel('Precision')
        ax[1][1].legend(loc=4)
    #logging.info(f"ROC and PR curve saved in {output_path}/roc_pr_curve_{model}_fold_{fold}.png")

    return round(accuracy, 5), round(accuracy_balanced, 5), round(f1_binary, 5), round(f1_weighted, 5), round(mcc, 5), round(auroc_class, 5), round(auprc_class, 5), round(auroc_proba, 5), round(auprc_proba, 5), ax
