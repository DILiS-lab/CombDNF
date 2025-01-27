# CombDNF
---
Drug combinations are increasingly applied to treat a wide range of complex diseases. Drug action and thus also drug combination effects can differ between diseases, e.g., due to molecular differences. Therefore, disease-specific predictions are required for treatments. A plethora of methods based on cell-line screening data in cancer has been proposed. However, their extendability to other diseases is limited, as is their applicability in the clinical context due to the in-vivo-in-vitro gap. In contrast, only few approaches rely on clinically validated data.
Here, we propose CombDNF, a novel machine-learning based method for disease-specific drug combination prediction on clinically validated data. CombDNF is trained and predicts both on clinically approved (effective) and clinically reported adverse drug combinations from a broad collection of data sources. It can cope with the highly imbalanced label distribution in drug combination data. Further, CombDNF leverages network-derived features based on drug target and disease gene relationships. To incorporate uncertainty of the network topology it relies on edge weights in the underlying network.
---

# Overview
CombDNF consists of two steps:
1. Network-based feature generation: Generate features for drug combination from protein-protein interaction network.
2. Drug Combination classification: Train and evaluate classification XGBoost and other classification models on drug combination data.

---
# 1. Network-based feature generation

Example code:
```python CombDNF_feature_generation_CLI.py -n PPI.txt -t drug_targets.txt -dm disease_genes.txt -o output/ -w -sp -c 4```

# 2. Drug Combination classification

Example code
```python CombDNF_classification_CLI.py -f features.tsv -g ground_truth.tsv -o output/ -p -pd both -pl 1 -c all -ba adasyn -t grid -sm matthews_corrcoef -ns 5 -r 123 -cpu 4```

