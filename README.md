# REMEDI: A Benchmark for Retention and Erasure Evaluation in Multi-label Clinical Disease Inference

This repository contains the code and benchmark data for the paper titled **"REMEDI: A Benchmark for Retention and Erasure Evaluation in Multi-label Clinical Disease Inference."** The paper is currently under review. We have provided sample benchmark data in this repository. The full benchmark preparation scripts will be released upon acceptance of the paper.

The benchmark is constructed using the MIMIC-III Clinical Database, which is a licensed resource. To fully utilize this benchmark, users must obtain access to the MIMIC-III Clinical Database and run the provided scripts to prepare the benchmark data.

# Demo Code
Results can be obtained using the main notebook (demo.ipynb)

The code is commented for ease of use.

| Method     | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ---------- | ------- | --------- | --------- | --------------- |
| Original   | 0.6042  | 0.6330    | 0.6314    | 0.0307          |
| Retrain    | 0.6291  | 0.6216    | 0.6128    | 0.0692          |
| NegGrad    | 0.0739  | 0.0710    | 0.0757    | 0.1076          |
| NegGrad+   | 0.6278  | 0.5906    | 0.6578    | 0.0076          |
| Adv        | 0.5210  | 0.4885    | 0.5538    | 0.0692          |
| Adv+Imp    | 0.5266  | 0.5079    | 0.5513    | 0.0076          |
| BadTeacher | 0.4717  | 0.4749    | 0.4906    | 0.1230          |
| SCRUB      | 0.6249  | 0.6673    | 0.6539    | 0.0153          |
