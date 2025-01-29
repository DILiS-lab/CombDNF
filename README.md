# CombDNF

Drug combinations are increasingly applied to treat a wide range of complex diseases. Drug action and thus also drug combination effects can differ between diseases, e.g., due to molecular differences. Therefore, disease-specific predictions are required for treatments. A plethora of methods based on cell-line screening data in cancer has been proposed. However, their extendability to other diseases is limited, as is their applicability in the clinical context due to the in-vivo-in-vitro gap. In contrast, only few approaches rely on clinically validated data.
Here, we propose CombDNF, a novel machine-learning based method for disease-specific drug combination prediction on clinically validated data. CombDNF is trained and predicts both on clinically approved (effective) and clinically reported adverse drug combinations from a broad collection of data sources. It can cope with the highly imbalanced label distribution in drug combination data. Further, CombDNF leverages network-derived features based on drug target and disease gene relationships. To incorporate uncertainty of the network topology it relies on edge weights in the underlying network.


# Overview
CombDNF consists of two steps:
1. Network-based feature generation: Generate features for drug combination from protein-protein interaction network.
2. Drug Combination classification: Train and evaluate classification XGBoost and other classification models on drug combination data.

## Requirements
CombDNF is implemented and tested in python 3.10.14. All package requirements are available in requirements.txt

# 1. Network-based feature generation

Example run:
```python CombDNF_feature_generation_CLI.py -n test_data/test_PPI_network.txt -t test_data/test_drug_targets.txt -dm test_data/test_disease_genes.txt -o test_data/output/ -w -sp -c 4```

For information on all parameters see
```python CombDNF_feature_generation_CLI.py -h ```

# 2. Drug Combination classification

Example run:
```python CombDNF_classification_CLI.py -f test_data/output/CombDNF_scores.tsv -g test_data/test_drug_combinations.tsv -o test_data/output/ -ba adasyn -t grid -sm matthews_corrcoef -ns 5 -pl 1 -r 123 -cpu 4```

For information on all parameters see
```python CombDNF_classification_CLI.py -h ```


# Data
We generated disease-specific drug combination features with CombDNF for the following diseases:

1. Neoplasms
2. Hypertension
3. Cardiovascular diseases
4. Nervous system diseases

All files are available in data/

## Ground truth data

Ground truth data was gathered from the following sources:
- Cheng et al. (2019) supplement: https://doi.org/10.1038/s41467-019-09186-x
- Das et al. (2019) supplement: https://doi.org/10.1021/acs.jmedchem.8b01610
- CDCDB (release date June 18, 2024): https://icc.ise.bgu.ac.il/medical_ai/CDCDB/
- DrugCombDB (release date May 31, 2019): http://drugcombdb.denglab.org/main
- DrugBank (release 5.1.11): https://go.drugbank.com/releases



