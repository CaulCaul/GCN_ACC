# GCN_ACC

### 1. How to simulate GCNAX

Run `simulator.py`

The function is implemented in file `model/GCNAX_simple_model.py`

### 2. How to simulate our model

Run `estimate/gen_data.py` to get the data file. (`cora_src.mat` about 160MB and `pubmed_src.mat` about 350MB)

Then run `estimate/cora_eval0.py`. Need to manually change the code to determine which dataset to read.