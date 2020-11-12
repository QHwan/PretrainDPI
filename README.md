# PretrainDPI

This code classifies binary drug-protein interaction data with the pretrained protein embedding and Bayesian neural networks.

At first, you need to convert raw data format into deep learning-ready. In the ./data folder, run preprocess.py
```python
python preprocess.py --dataset human --n_split 10 --pretrained transformer12
```
We use pretrained model from Rives et al.(BioRxiv 622803; doi:https://doi.org/10.1101/622803). The model is available in GitHub repository (github.com/facebookresearch/esm). If you don't have model, preprocess.py code automatically download it.  

We support two training codes, main_nn.py and main_dropout.py. The latter file uses Bayesian training with MC-dropout method. We adopt concrete dropout.
```python
python main_dropout.py --dataset_file {dataset_path} --save_model {path_to_save_your_trained_model} --save_result {path_to_save_your_classification_result} 
```
