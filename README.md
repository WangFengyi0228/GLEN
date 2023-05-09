# GLEN
This is the code associated with the submission "Understanding Temporal Graph Learning From Global and Local Perspectives".

Our code references the benchmark [DGB](https://github.com/fpour/DGB).

## Running the experiments


### Dependencies (with python >= 3.9):

```{bash}
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c anaconda pandas
conda install -c anaconda scikit-learn
conda install -c conda-forge tqdm
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install pyg -c pyg
```

### Datasets

The folder 'data' contains processed datasets. All the datasets used in the paper can be downloaded from [here](https://zenodo.org/record/7213796#.ZClGKXZBxD9).

### Run scripts (model training and evaluation)

#### Link prediction

To run GLEN on Wikipedia, Reddit, UCI, Enron, MOOC, UN Trade for link prediction task:

```{bash}
python train_link_prediction.py -d wikipedia --n_degree 10 --n_head 2 --drop_out 0.3 --window 2 --n_runs 5
python train_link_prediction.py -d reddit --n_degree 10 --n_head 4 --drop_out 0.3 --n_runs 5 --window 2 --n_runs 5
python train_link_prediction.py -d uci --n_degree 30 --n_head 1 --drop_out 0.5 --window 2 --n_runs 5
python train_link_prediction.py -d enron --n_degree 20 --n_head 4 --drop_out 0.1 --window 1 --n_runs 5
python train_link_prediction.py -d UNtrade --n_degree 20 --n_head 1 --drop_out 0.1 --window 1 --n_runs 5
python train_link_prediction.py -d mooc --n_degree 20 --n_head 4 --drop_out 0.1 --window 8 --n_runs 5
```

Apply DGB test for GLEN ([DGB repository](https://github.com/fpour/DGB)):


```{bash}
python dgb_test_for_GLEN.py -d wikipedia --n_degree 10 --n_head 2 --drop_out 0.3 --window 2 --neg_sample rnd --n_runs 5
python dgb_test_for_GLEN.py -d wikipedia --n_degree 10 --n_head 2 --drop_out 0.3 --window 2 --neg_sample hist_nre --n_runs 5
python dgb_test_for_GLEN.py -d wikipedia --n_degree 10 --n_head 2 --drop_out 0.3 --window 2 --neg_sample induc_nre --n_runs 5

python dgb_test_for_GLEN.py -d reddit --n_degree 10 --n_head 4 --drop_out 0.3 --window 2 --neg_sample rnd --n_runs 5
python dgb_test_for_GLEN.py -d reddit --n_degree 10 --n_head 4 --drop_out 0.3 --window 2 --neg_sample hist_nre --n_runs 5
python dgb_test_for_GLEN.py -d reddit --n_degree 10 --n_head 4 --drop_out 0.3 --window 2 --neg_sample induc_nre --n_runs 5

python dgb_test_for_GLEN.py -d uci --n_degree 30 --n_head 1 --drop_out 0.5 --window 2 --neg_sample rnd --n_runs 5
python dgb_test_for_GLEN.py -d uci --n_degree 30 --n_head 1 --drop_out 0.5 --window 2 --neg_sample hist_nre --n_runs 5
python dgb_test_for_GLEN.py -d uci --n_degree 30 --n_head 1 --drop_out 0.5 --window 2 --neg_sample induc_nre --n_runs 5

python dgb_test_for_GLEN.py -d enron --n_degree 20 --n_head 4 --drop_out 0.1 --window 1 --neg_sample rnd --n_runs 5
python dgb_test_for_GLEN.py -d enron --n_degree 20 --n_head 4 --drop_out 0.1 --window 1 --neg_sample hist_nre --n_runs 5
python dgb_test_for_GLEN.py -d enron --n_degree 20 --n_head 4 --drop_out 0.1 --window 1 --neg_sample induc_nre --n_runs 5

python dgb_test_for_GLEN.py -d UNtrade --n_degree 20 --n_head 1 --drop_out 0.1 --window 1 --neg_sample rnd --n_runs 5
python dgb_test_for_GLEN.py -d UNtrade --n_degree 20 --n_head 1 --drop_out 0.1 --window 1 --neg_sample hist_nre --n_runs 5
python dgb_test_for_GLEN.py -d UNtrade --n_degree 20 --n_head 1 --drop_out 0.1 --window 1 --neg_sample induc_nre --n_runs 5

python dgb_test_for_GLEN.py -d mooc --n_degree 20 --n_head 4 --drop_out 0.1 --window 8 --neg_sample rnd --n_runs 5
python dgb_test_for_GLEN.py -d mooc --n_degree 20 --n_head 4 --drop_out 0.1 --window 8 --neg_sample hist_nre --n_runs 5
python dgb_test_for_GLEN.py -d mooc --n_degree 20 --n_head 4 --drop_out 0.1 --window 8 --neg_sample induc_nre --n_runs 5
```

**rnd** means random negative sampling.

**hist_nre** means historical negative sampling.

**induc_nre** means inductive negative sampling.

#### Dynamic node classification

To run GLEN on Wikipedia, Reddit, MOOC for dynamic node classification task:

```{bash}
python pretrain_for_nc.py -d wikipedia --n_degree 10 --n_head 4 --drop_out 0.3 --window 1
python train_node_classifiction.py -d wikipedia --n_degree 10 --window 1 --n_runs 5

python pretrain_for_nc.py -d reddit --n_degree 10 --n_head 4 --drop_out 0.3 --window 2
python train_node_classifiction.py -d reddit --n_degree 10 --window 2 --n_runs 5

python pretrain_for_nc.py -d mooc --n_degree 10 --n_head 4 --drop_out 0.3 --window 8
python train_node_classifiction.py -d mooc --n_degree 10 --window 8 --n_runs 5
```

