# PretrainDPI

This code classifies binary drug-protein interaction data with the pretrained protein embedding and Bayesian neural networks.

At first, you need to convert raw data format into deep learning-ready. In the ./data folder, run preprocess.py
```python
python preprocess.py --dataset human --n_split 10 --pretrained transformer12
```
