Environment:
tensorflow==2.4.0
neurokit2==0.1.7
scipy==1.4.1
scikit-learn==0.24.2
matplotlib==3.3.4
numpy==1.17.0

1. shao.py/ptb.py/ptbxl.py: load data from 3 datasets
2. limb.data_preprocess.py: perform preprocess on signals
3. limb.LDenseNet.py: build LDenseNet model
4. limb.train_model_limb.py: make limb lead misplacement dataset, train and evaluate the LDenseNet 
5. chest.get_dataset: make chest lead misplacement dataset
6. chest.ILC_model: build ILC model
7. chest.train_model_chest: train and evaluate the ILC model

An example of prediction can be found in prediciton_example.py
Samples are chosen from PTBXL, with the shape of (10,5000,12). Noise reduction, downsampling and Z-score normalization are carried out successively.
The input shape of LDenseNet is (N, 1200, 3), where 3 presents Ⅲ, avR, and V6. The input shape of ILC is (N, 1200, 6), where 6 presents 6 chest leads.
Model's weights are saved at 'saved_model' floder.
