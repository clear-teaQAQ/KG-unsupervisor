# GEDRanker

This repository is the official implementation of NeurIPS 2025 paper "Towards Unsupervised Training of Matching-based
Graph Edit Distance Solver via Preference-aware GAN".

## Get Started
Please install packages as specified in `requirements.txt`.

The implementation of GEDRanker and other baseline models can be founded in `src/`.

### Datasets
The datasets [AIDS, Linux, IMDB] in `json_data/` are obtained from [[GEDGNN]](https://github.com/ChengzhiPiao/GEDGNN).

### Training & Evaluation

To train GEDRanker on AIDS:
```
python GEDRanker/main.py --dataset AIDS --model-epoch-start 0 --model-epoch-end 200 --model-train 1
```
Trained models are saved in `model_save/`.

To evaluate GEDRanker on AIDS:
```
python GEDRanker/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 
```

Results are saved in `result/`.

To train and evaluate GEDRanker on other datasets, replace `AIDS` of the `dataset` parameter with `Linux` or `IMDB`.


To run GEDRanker on datasets that do not have ground-truth labels for evaluation (e.g., IMDB-large):
```
python GEDRanker/main_no_gt.py --dataset IMDB --model-epoch-start 0 --model-epoch-end 200 --model-train 1 
```
```
python GEDRanker/main_no_gt.py --dataset IMDB --model-epoch-start 200 --model-epoch-end 200 --model-train 0 
```



### Baseline methods
To train and evaluate each baseline method, follow the instructions in each baseline's directory within `src/`.


