# PretrainDPI

This code classifies binary drug-protein interaction data with the pretrained protein embedding and Bayesian neural networks.

At first, you need to convert raw data format into deep learning-ready. In the ./data folder, run preprocess.py
```python
python preprocess.py --dataset human --n_split 10 --pretrained transformer12
```
We use pretrained model from Rives et al.(BioRxiv 622803; doi:https://doi.org/10.1101/622803). The model is available in GitHub repository (github.com/facebookresearch/esm). If you don't have model, preprocess.py code automatically download it.  
