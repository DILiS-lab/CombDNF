# CombDNF

Drug combinations are increasingly applied to treat a wide range of complex diseases. Drug action and thus also drug combination effects can differ between diseases, e.g., due to molecular differences. Therefore, disease-specific predictions are required for treatments. A plethora of methods based on cell-line screening data in cancer has been proposed. However, their extendability to other diseases is limited, as is their applicability in the clinical context due to the in-vivo-in-vitro gap. In contrast, only few approaches rely on clinically validated data.

Here, we propose CombDNF, a novel machine-learning based method for disease-specific drug combination prediction on clinically validated data. CombDNF is trained and predicts both on clinically approved (effective) and clinically reported adverse drug combinations from a broad collection of data sources. It can cope with the highly imbalanced label distribution in drug combination data. Further, CombDNF leverages network-derived features based on drug target and disease gene relationships. To incorporate uncertainty of the network topology it relies on edge weights in the underlying network.


# Overview

CombDNF consists of two steps:
1. Network-based feature generation: Generate features for drug combination from protein-protein interaction network.
2. Drug Combination classification: Train and evaluate classification XGBoost and other classification models on drug combination data.

## Requirements

CombDNF is implemented and tested in python 3.10.14. All package requirements are available in requirements.txt

## Exemplary data
See test_data/ for a small test set for testing CombDNF. This also shows the input data structure used in CombDNF.

## 1. Network-based feature generation
Network-based features are generated using a protein-protein interaction network, drug-target interactions and disease genes as input. Features can be computed on the binary network or weighted network (with ```-w```)

Example run:
```python CombDNF_feature_generation_CLI.py -n test_data/test_PPI_network.txt -t test_data/test_drug_targets.txt -dm test_data/test_disease_genes.txt -o test_data/output/ -w -sp -c 4```

For information on all parameters see
```python CombDNF_feature_generation_CLI.py -h```

### Input data description
Input data for CombDNF feature generation consists of three files:
1. Protein-protein interaction network: A tab-separated file with two columns containing the interacting proteins. The first column contains the source protein and the second column contains the target protein.
2. Drug target file: A tab-separated file with two columns containing the drug and the target protein. The first column contains the drug and the second column contains the target protein.
3. Disease module file: A tab-separated file with one column containing the disease proteins.

### Output data description
The output of CombDNF feature generation is a folder containing the following files:
1. CombDNF_feature_generation.log: A log file containing the information about the feature generation process.
(2.) CombDNF_unweighted_shortest_path_lengths.tsv/CombDNF_weighted_shortest_path_lengths.tsv: A tab-separated file with all shortest paths lengths between all nodes in the network.
3.CombDNF_drug_drug_scores.tsv: A tab-separated file with eight columns containing the drug-drug pairs and their distance scores.
4. CombDNF_drug_disease_scores.tsv: A tab-separated file with nine columns containing the drug-disease pairs and their distance scores.
5. CombDNF_scores.tsv: A tab-separated file with 22 features for each drug-drug and drug-disease pair.

## 2. Drug combination classification
Drug combinations classification uses features and ground truth data input for training and testing XGBoost models for classification.

Example run:
```python CombDNF_classification_CLI.py -f test_data/output/CombDNF_scores.tsv -g test_data/test_drug_combinations.tsv -o test_data/output/ -ba adasyn -t grid -sm matthews_corrcoef -ns 5 -cpu 4```

For information on all parameters see
```python CombDNF_classification_CLI.py -h```


### Input data description
Input data for CombDNF classification should be tab separated files with features and ground truth data.
1. Features data: tab separated file with features for drug combinations. Required columns are 'drugA' and 'drugB' for drug identifier. The remaining columns are considered as features.
2. Ground truth data: tab separated file with ground truth data for drug combinations. Required columns are 'drugA', 'drugB' and 'label'. The 'label' column should contain the binary label for drug combinations. The 'label' column should contain the positive label for approved drug combinations.
3. Output path: path to the folder in which the output files should be saved.

### Output data description
1. CombDNF_classification.log: log file with all information and errors during the classification process.
2. CombDNF_predictions_test_sets.csv: tab separated file with predictions for test sets of drug combinations. The file contains the original drug combination labels, the predicted labels and probabilities for all classification models.
3. CombDNF_predictions_test_sets_scores.csv: tab separated file with scores for test sets of drug combinations. The file contains the scores for all classification models and cross-validation folds.
4. CombDNF_predictions_new_combinations.csv: tab separated file with predictions for new drug combinations. The file contains the predicted labels and probabilities for all classification models.
5. CombDNF_hyperparameter_tuning_scores_{model}.csv: tab separated file with scores and hyperparameters for all tested hyperparameters of the classification model.
6. CombDNF_best_model_{model}_fold{i}.joblib: joblib file with the best model for each fold of the cross-validation.
7. roc_pr_curve_{model}.png: figure with ROC and PR curves for the classification model.


We generated disease-specific drug combination features with CombDNF for the following diseases:

1. Neoplasms
2. Hypertension
3. Cardiovascular diseases
4. Nervous system diseases

All files are available in data/. We predict the drug combinations for these diseases using the generated features and the ground truth data from the following sources:

## Ground truth data

Ground truth data was gathered from the following sources:

Effective drug combinations:
- Cheng et al. (2019) supplement: https://doi.org/10.1038/s41467-019-09186-x
- Das et al. (2019) supplement: https://doi.org/10.1021/acs.jmedchem.8b01610
- CDCDB (release date June 18, 2024): https://icc.ise.bgu.ac.il/medical_ai/CDCDB/
- DrugCombDB (release date May 31, 2019): http://drugcombdb.denglab.org/main
- DrugBank (release 5.1.11): https://go.drugbank.com/releases

Adverse drug combinations:
- Cheng et al. (2019) supplement: https://doi.org/10.1038/s41467-019-09186-x
- DrugBank (release 5.1.11): https://go.drugbank.com/releases



