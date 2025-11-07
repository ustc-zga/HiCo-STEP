### HiCo-STEP: Causality Informed Hierarchical Spatiotemporal Network for Traffic Emission Prediction  ####


1. Install Pytorch and necessary dependencies.
## Requirements:
We build this project by Python 3.9.19 with the following packages: 
einops==0.6.0
numpy==1.24.1
omegaconf==2.3.0
scipy==1.9.1
torch==2.1.0

```
pip install -r requirements.txt


2. Datasets and preprocess
--Step 1----------
We provide the datasets of Xian and Beijing in the paper and you can download them from [https://pan.baidu.com/s/1A9TOxjCaTai0qsXhx1sL4Q].(Extraction code: s2x9). The files contains the original traffic emission data and conresponding  adjacent matrix files. Unzip the datasets and place them in the data folder [./data].
|----data\
|    |----xianCO_12.npy  
|    |----adj_12.npy         
```

--Step 2----------
Run the data preprocess script [preprocess_data.py] and place the output into the folder [./data/XiAn_City].

Each dataset is composed of 3 files, namely `train.npz`, `val.npz`, `test.npz`
```
|----XiAn_City\
|    |----train.npz    
|    |----test.npz    
|    |----val.npz      
```

#### Model training
 
The hypapameter used in our paper is contained in [./config.yaml].
To train the model, you can run 
---------
python main.py 




