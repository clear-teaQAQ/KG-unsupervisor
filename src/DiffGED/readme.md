# DiffGED

This repository is the implementation of paper "DiffGED: Solving Graph Edit Distance via Diffusion-based Graph Matching", which aims to solve graph edit distance (GED) based on generative diffusion-based graph matching model.


### Training & Evaluation

To train DiffGED on AIDS:
```
python DiffGED/main.py --dataset AIDS --model-epoch-start 0 --model-epoch-end 200 --model-train 1
```
Trained models are saved in `model_save/`.

To evaluate DiffGED on AIDS with k = 100:
```
python DiffGED/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --test-k 100 --topk-approach parallel
```

Results are saved in `result/`.

To train and evaluate DiffGED on other datasets, replace `AIDS` of the `dataset` parameter with `Linux` or `IMDB`.

### Experiments
To evaluate DiffGED top-k approach on effectiveness and efficiency:
```
python DiffGED/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --experiment topk_analysis --topk-approach parallel
```
To evaluate GEDGNN top-k approach (DiffGED-single) on effectiveness and efficiency:
```
python DiffGED/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --experiment topk_analysis --topk-approach sequential
```
To evaluate DiffGED top-k approach on edit paths diversity:
```
python DiffGED/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --experiment diversity_analysis --topk-approach parallel --test-k 100
```
To evaluate GEDGNN top-k approach (DiffGED-single) on edit paths diversity:
```
python DiffGED/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --experiment diversity_analysis --topk-approach sequential --test-k 100
```



