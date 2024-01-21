# MGTBench

MGTBench provides the reference implementations of different machine-generated text (MGT) detection methods.
It is still under continuous development and we will include more detection methods as well as analysis tools in the future.


## Supported Methods
Currently, we support the following methods (continuous updating):
- Metric-based methods:
    - Log-Likelihood [[Ref]](https://arxiv.org/abs/1908.09203);
    - Rank [[Ref]](https://arxiv.org/abs/1906.04043);
    - Log-Rank [[Ref]](https://arxiv.org/abs/2301.11305);
    - Entropy [[Ref]](https://arxiv.org/abs/1906.04043);
    - GLTR Test 2 Features (Rank Counting) [[Ref]](https://arxiv.org/abs/1906.04043);
    - DetectGPT [[Ref]](https://arxiv.org/abs/2301.11305);
    - LRR [[Ref]](https://arxiv.org/abs/2306.05540);
    - NPR [[Ref]](https://arxiv.org/abs/2306.05540);
- Model-based methods:
    - OpenAI Detector [[Ref]](https://arxiv.org/abs/1908.09203);
    - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
    - ConDA [[Ref]](https://arxiv.org/abs/2309.03992) [[Model Weights]](https://www.dropbox.com/s/sgwiucl1x7p7xsx/fair_wmt19_chatgpt_syn_rep_loss1.pt?dl=0);
    - GPTZero [[Ref]](https://gptzero.me/);
    - LM Detector [[Ref]](https://arxiv.org/abs/1911.00650);

## Supported Datasets
- Essay;
- WP;
- Reuters; 

Note that our datasets are constructed based on [Verma et al.](https://arxiv.org/abs/2305.15047), you can download them from [Google Drive](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola?usp=sharing).

## Installation
```
git clone https://github.com/xinleihe/MGTBench.git;
cd MGTBench;
conda env create -f environment.yml;
conda activate MGTBench;
```

## Usage
To run the benchmark on the Essay dataset: 
```
# Distinguish Human vs. Claude:
python benchmark.py --dataset Essay --detectLLM Claude --method Log-Likelihood

# Text attribution:
python attribution_benchmark.py --dataset Essay
```
Note that you can also specify your own datasets on ``dataset_loader.py``.


## Authors
The tool is designed and developed by Xinlei He (CISPA), Xinyue Shen (CISPA), Zeyuan Chen (Individual Researcher), Michael Backes (CISPA), and Yang Zhang (CISPA).

## Cite
If you use MGTBench for your research, please cite [MGTBench: Benchmarking Machine-Generated Text Detection](https://arxiv.org/abs/2303.14822).

```
bibtex
@article{HSCBZ23,
author = {Xinlei He and Xinyue Shen and Zeyuan Chen and Michael Backes and Yang Zhang},
title = {{MGTBench: Benchmarking Machine-Generated Text Detection}},
journal = {{CoRR abs/2303.14822}},
year = {2023}
}
```
