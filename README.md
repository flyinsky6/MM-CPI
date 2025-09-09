# MM-CPI
MM-CPI, a novel and effective multimodal deep learning model for predicting compound–protein interactions (CPI), which integrates molecular fingerprints, compound language representations, and molecular graph structures via Graph Convolutional Networks with protein features derived from sequence information, physicochemical properties, and protein language models. Then, within a deep learning framework, feature fusion and interaction modeling were conducted to enable accurate classification, and experimental evaluations on multiple public datasets demonstrated that MM-CPI significantly outperformed existing methods in accuracy, robustness, and generalization. K-fold cross-validation, independent testing, and a Fabry disease case study confirmed that MM-CPI can serve as a powerful tool for CPI prediction and drug repurposing.
# Requirement
keras(2.3.1)
Numpy(1.2.4)
scikit-learn(1.0.2)
pandas(1.3.5)
matplotlib(3.5.2)
Mol2vec(0.2.2)
Python(3.8.2)
rdkit-pypi(2022.9.4) 
scipy(1.10.1)
seaboen(0.13.2)
torch（2.4.1）
transformers（4.46.3）
# Dataset
This study employed four benchmark datasets from CPI research for model training and testing. All datasets were randomly partitioned into training, validation, and test sets at a ratio of 7:1:2.
# Model
The best model of different feature can be downloaded in model directory.
# Contact
Feel free to contact us if you nedd any help: lzq.work@foxmail.com
