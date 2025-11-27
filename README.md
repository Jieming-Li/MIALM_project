# MIALM_project
Class project for Privacy Aware Computing

Strategy:
Shadow models are trained on shadow pool (sampled.csv) to generate data to train attacker model.
Output signal used: softmax probabilities
This is better than using validation dataset to train attacker or simple threshold method since validation dataset is only a small dataset.

