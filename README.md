# MGTBench

MGTBench provides the referece implementations of different machine-generated text (MGT) detection methods.
It is under continuous development and we will include more detection methods as well as analysis tools in the future.


## Suopported Methods
Currently, we support the following methods:
- Metric-based methods:
    - Log-Likelihood;
    - Rank;
    - Log-Rank;
    - Entropy;
    - GLTR Test 2 Features (Rank Counting);
    - DetectGPT;
- Model-based methods:
    - Openai Detector;
    - ChatGPT Detector;

## Suopported Datasets
- TruthfulQA;
- SQuAD1;
- NarrativeQA; (For NarrativeQA, you can download the dataset from [Google Drive](https://drive.google.com/file/d/1tul8WeWqubyRlxLeJ5L3igfaL6VNSuef/view?usp=share_link).)

## Usage
To run the benchmark on the SQuAD1 dataset: 
```
python benchmark.py --dataset SQuAD1 --base_model_name gpt2-medium --mask_filling_model_name t5-large
```

Note that you can also specify your own datasets on ``dataset_loader.py``

## Authors
The tool is designed and developed by Xinlei He (CISPA), Xinyue Shen (CISPA), Zeyuan Chen (Individual Researcher), Michael Backes (CISPA), and Yang Zhang (CISPA).