
# Causal Discovery with Local A*

This repository contains an implementation of the causal discovery/structure learning method described in ["Reliable Causal Discovery with Improved Exact Search and Weaker Assumptions"](https://papers.nips.cc/paper/2021/hash/a9b4ec2eb4ab7b1b9c3392bb5388119d-Abstract.html). 

If you find it useful, please consider citing:
```bibtex
@inproceedings{Ng2021reliable,
  author = {Ignavier Ng and Yujia Zheng and Jiji Zhang and Kun Zhang},
  title = {Reliable Causal Discovery with Improved Exact Search and Weaker Assumptions},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2021},
}
```

## Requirements

Python 3.6+ is required. To install the requirements:
```setup
pip install -r requirements.txt
```

## Running A* with Super-Structure
- Each run creates a directory based on current datetime to save the training outputs.
- For graphs with more than 20 variables, set `glasso_l1` to `0.2`.
- For graphs with more than 40 variables, set `glasso_thres` to `0.03`.
- One may set `search_method` to `dp` to use dynamic programming instead of A*.
- If needed, one may further use suitable model selection methods (e.g., cross-validation) to select the hyperparameters.
```
# Ground truth: 10-variable Erdos–Renyi graph with expected degree of 2
# Data: Linear Gaussian model with 10000 samples
# Super-structure: Support of estimated inverse covariance matrix ussing graphical Lasso
python src/main.py  --seed 1 \
                    --d 10 \
                    --n 10000 \
                    --degree 2 \
                    --super_graph_method glasso \
                    --search_strategy global \
                    --search_method astar \
                    --glasso_l1 0.05 \
                    --glasso_thres 0.0 \
                    --n_jobs -1 \
                    --use_path_extension \
                    --use_k_cycle_heuristic \
                    --k 3 \
                    --verbose
```

## Running Local A*
- Each run creates a directory based on current datetime to save the training outputs.
- For graphs with more than 20 variables, set `glasso_l1` to `0.2`.
- For graphs with more than 40 variables, set `glasso_thres` to `0.03`.
- One may set `search_method` to `dp` to use dynamic programming instead of A*.
- If needed, one may further use suitable model selection methods (e.g., cross-validation) to select the hyperparameters.
```
# Ground truth: 10-variable Erdos–Renyi graph with expected degree of 2
# Data: Linear Gaussian model with 10000 samples
# Super-structure: Support of estimated inverse covariance matrix ussing graphical Lasso
python src/main.py  --seed 1 \
                    --d 10 \
                    --n 10000 \
                    --degree 2 \
                    --super_graph_method glasso \
                    --search_strategy local \
                    --search_method astar \
                    --glasso_l1 0.05 \
                    --glasso_thres 0.0 \
                    --n_jobs -1 \
                    --use_path_extension \
                    --use_k_cycle_heuristic \
                    --k 3 \
                    --verbose
```

## Acknowledgments
- The code to  compute the metrics (e.g., SHD, TPR) is based on [NOTEARS](https://github.com/xunzheng/notears/blob/master/notears/utils.py).
- The code to log the experiment outputs is based on [GOLEM](https://github.com/ignavierng/golem).
- The code to generate the synthetic data is based on [NOTEARS](https://github.com/xunzheng/notears/blob/master/notears/utils.py) and [GOLEM](https://github.com/ignavierng/golem/blob/main/src/data_loader/synthetic_dataset.py).
- We are grateful to the authors of the baseline methods for releasing their code.