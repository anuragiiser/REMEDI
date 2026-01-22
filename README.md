# REMEDI: A Benchmark for Retention and Erasure Evaluation in Multi-label Clinical Disease Inference

This repository contains the code and benchmark data for the paper titled **"REMEDI: A Benchmark for Retention and Erasure Evaluation in Multi-label Clinical Disease Inference."** The paper is currently under review. We have provided sample benchmark data in this repository. The full benchmark preparation scripts will be released upon acceptance of the paper.

The benchmark is constructed using the MIMIC-III Clinical Database, which is a licensed resource. To fully utilize this benchmark, users must obtain access to the MIMIC-III Clinical Database and run the provided scripts to prepare the benchmark data.

# Demo Code
Results can be obtained using the main notebook (demo.ipynb)

The code is commented for ease of use.

# Results

## Distinct forget sets

K = 64

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

K = 128

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


K = 256 

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

## Large-scale forget set

K = 3%

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.5388  | 0.5631    | 0.5692    | 0.0560          |
| Retrain     | 0.5522  | 0.5407    | 0.5564    | 0.0103          |
| NegGrad     | 0.0216  | 0.0221    | 0.0212    | 0.0307          |
| NegGrad+    | 0.1961  | 0.1800    | 0.2051    | 0.0240          |
| Adv         | 0.0216  | 0.0221    | 0.0213    | 0.0247          |
| Adv+Imp     | 0.0218  | 0.0221    | 0.0214    | 0.0010          |
| Bad Teacher | 0.1431  | 0.1479    | 0.1504    | 0.0313          |
| SCRUB       | 0.5674  | 0.5782    | 0.5852    | 0.0362          |

K = 4%

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.5388  | 0.5682    | 0.5691    | 0.0384          |
| Retrain     | 0.5480  | 0.5434    | 0.5480    | 0.0114          |
| NegGrad     | 0.0216  | 0.0216    | 0.0213    | 0.0057          |
| NegGrad+    | 0.2042  | 0.1881    | 0.2131    | 0.0351          |
| Adv         | 0.0216  | 0.0216    | 0.0213    | 0.0074          |
| Adv+Imp     | 0.0217  | 0.0216    | 0.0213    | 0.0363          |
| Bad Teacher | 0.1470  | 0.1499    | 0.1544    | 0.0024          |
| SCRUB       | 0.5796  | 0.5912    | 0.5997    | 0.0252          |

K = 5%

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.5388  | 0.5703    | 0.5690    | 0.0241          |
| Retrain     | 0.5320  | 0.5230    | 0.5340    | 0.0142          |
| NegGrad     | 0.0216  | 0.0215    | 0.0213    | 0.0129          |
| NegGrad+    | 0.2033  | 0.1900    | 0.2138    | 0.0347          |
| Adv         | 0.0216  | 0.0215    | 0.0213    | 0.0135          |
| Adv+Imp     | 0.0217  | 0.0215    | 0.0213    | 0.0003          |
| Bad Teacher | 0.1344  | 0.1424    | 0.1413    | 0.0019          |
| SCRUB       | 0.5904  | 0.6181    | 0.6180    | 0.0278          |

## Concurrent forget set

K = 64

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.5388  | 0.5665    | 0.5690    | 0.0076          |
| Retrain     | 0.5811  | 0.5459    | 0.5808    | 0.0230          |
| NegGrad     | 0.0598  | 0.0595    | 0.0611    | 0.0846          |
| NegGrad+    | 0.5670  | 0.5067    | 0.5974    | 0.0307          |
| Adv         | 0.2823  | 0.2850    | 0.3216    | 0.0538          |
| Adv+Imp     | 0.3207  | 0.3086    | 0.3555    | 0.0307          |
| Bad Teacher | 0.4451  | 0.4575    | 0.4728    | 0.1076          |
| SCRUB       | 0.5564  | 0.5810    | 0.5860    | 0.1000          |

K = 128

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.5388  | 0.5849    | 0.5690    | 0.0576          |
| Retrain     | 0.5557  | 0.5389    | 0.5576    | 0.0384          |
| NegGrad     | 0.0320  | 0.0283    | 0.0314    | 0.1000          |
| NegGrad+    | 0.5881  | 0.5450    | 0.6166    | 0.0346          |
| Adv         | 0.1449  | 0.1541    | 0.1765    | 0.0384          |
| Adv+Imp     | 0.1916  | 0.2135    | 0.2278    | 0.0269          |
| Bad Teacher | 0.4399  | 0.4557    | 0.4674    | 0.0653          |
| SCRUB       | 0.5741  | 0.6111    | 0.5944    | 0.0423          |

K = 256

| Method      | F1 Test | F1 Forget | F1 Retain | Privacy Leakage |
| ----------- | ------- | --------- | --------- | --------------- |
| Original    | 0.5388  | 0.5805    | 0.5689    | 0.0596          |
| Retrain     | 0.5718  | 0.5693    | 0.5767    | 0.0570          |
| NegGrad     | 0.0473  | 0.0446    | 0.0455    | 0.0384          |
| NegGrad+    | 0.6021  | 0.5620    | 0.6371    | 0.0384          |
| Adv         | 0.0676  | 0.0668    | 0.0699    | 0.0384          |
| Adv+Imp     | 0.0757  | 0.0855    | 0.0861    | 0.0576          |
| Bad Teacher | 0.4246  | 0.4147    | 0.4488    | 0.0250          |
| SCRUB       | 0.5797  | 0.6086    | 0.6036    | 0.0538          |


