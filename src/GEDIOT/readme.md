# GEDIOT

To train GEDIOT on AIDS:
```
python GEDIOT/main.py --dataset AIDS --model-epoch-start 0 --model-epoch-end 200 --model-train 1 --model-name GEDIOT
```
To evaluate GEDIOT on AIDS:
```
python GEDIOT/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --model-name GEDIOT
```
To train and evaluate GEDIOT on other datasets, replace `AIDS` of the `dataset` parameter with `Linux` or `IMDB`.

To evaluate GEDGW on AIDS:
```
python GEDIOT/main.py --dataset AIDS  --model-name GEDGW
```

To evaluate GEDGW without ground-truth labels:
```
python GEDIOT/main_no_gt.py --dataset IMDB  --model-name GEDGW
```
