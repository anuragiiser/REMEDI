# REMEDI: A Benchmark for Retention and Erasure Evaluation in Multi-label Clinical Disease Inference

This repository contains the code and benchmark data for the paper titled **"REMEDI: A Benchmark for Retention and Erasure Evaluation in Multi-label Clinical Disease Inference."** The paper is currently under review. We have provided sample benchmark data in this repository. The full benchmark preparation scripts will be released upon acceptance of the paper.

The benchmark is constructed using the MIMIC-III Clinical Database, which is a licensed resource. To fully utilize this benchmark, users must obtain access to the MIMIC-III Clinical Database and run the provided scripts to prepare the benchmark data.

# Demo Code
Results can be obtained using the main notebook (demo.ipynb)

The code is commented for ease of use.

# Results

## Distinct Forget sets

K = 64 (Distinct)

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

K = 128 (Distinct)

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.6042  | 0.6444    | 0.6313    | 0.0615          |
| Retrain     | 0.5911  | 0.6014    | 0.6046    | 0.0499          |
| NegGrad     | 0.0547  | 0.0559    | 0.0545    | 0.1115          |
| NegGrad+    | 0.6322  | 0.5984    | 0.6595    | 0.0307          |
| Adv         | 0.4723  | 0.4467    | 0.5059    | 0.0346          |
| Adv+Imp     | 0.5096  | 0.5080    | 0.5397    | 0.0346          |
| Bad Teacher | 0.4815  | 0.5058    | 0.5019    | 0.0384          |
| SCRUB       | 0.6244  | 0.6606    | 0.6501    | 0.0730          |


K = 256 (Distinct)

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.6042  | 0.6428    | 0.6313    | 0.0403          |
| Retrain     | 0.5815  | 0.6011    | 0.6065    | 0.0154          |
| NegGrad     | 0.0633  | 0.0643    | 0.0626    | 0.0365          |
| NegGrad+    | 0.6361  | 0.5983    | 0.6680    | 0.0500          |
| Adv         | 0.1979  | 0.2047    | 0.2400    | 0.0250          |
| Adv+Imp     | 0.4041  | 0.4031    | 0.4350    | 0.0557          |
| Bad Teacher | 0.4788  | 0.4720    | 0.4961    | 0.0173          |
| SCRUB       | 0.6229  | 0.6480    | 0.6486    | 0.0019          |

K = 512 (Distinct)

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.6042  | 0.6215    | 0.6316    | 0.0524          |
| Retrain     | 0.5790  | 0.5982    | 0.5924    | 0.0203          |
| NegGrad     | 0.0484  | 0.0578    | 0.0472    | 0.0029          |
| NegGrad+    | 0.6296  | 0.5923    | 0.6584    | 0.0572          |
| Adv         | 0.0168  | 0.0179    | 0.0199    | 0.0135          |
| Adv+Imp     | 0.1638  | 0.2010    | 0.2180    | 0.0126          |
| Bad Teacher | 0.4798  | 0.4855    | 0.4974    | 0.0271          |
| SCRUB       | 0.5198  | 0.5226    | 0.5341    | 0.0165          |
