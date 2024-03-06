<p align="center">
<img src="pic/FedHCDR-Framework.png" width="760" height="450">
</p>

<div align="center">

# FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling
*[Hongyu Zhang](https://orion-orion.github.io/), Dongyi Zheng, Lin Zhong, Xu Yang, Jiyuan Feng, Yunqing Feng, [Qing Liao](http://liaoqing.hitsz.edu.cn/)\**

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/FedHCDR)[![LICENSE](https://img.shields.io/github/license/orion-orion/FedHCDR)](https://github.com/orion-orion/FedHCDR/blob/main/LICENSE)[![FedHCDR](https://img.shields.io/github/stars/orion-orion/FedHCDR)](https://github.com/orion-orion/FedHCDR)
<br/>
[![FedHCDR](https://img.shields.io/github/directory-file-count/orion-orion/FedHCDR)](https://github.com/orion-orion/FedHCDR) [![FedHCDR](https://img.shields.io/github/languages/code-size/orion-orion/FedHCDR)](https://github.com/orion-orion/FedHCDR)
</div>

## 1 Introduction

This is the source code and baselines of our paper *[FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal
Decoupling](https://arxiv.org/abs/2403.02630)*. In this paper, we propose **FedHCDR**, a novel federated cross-domain recommendation framework with hypergraph signal decoupling. 

## 2 Dependencies

Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```
Note that my Python version is `3.8.16`.

## 3 Dataset

We utilize publicly available datasets from the [Amazon](https://jmcauley.ucsd.edu/data/amazon/}{https://jmcauley.ucsd.edu/data/amazon/) website to construct FedCDR scenarios. We select ten domains to generate three cross-domain scenarios: Food-Kitchen-Cloth-Beauty (FKCB), Sports-Clothing-Elec-Cell (SCEC), and Sports-Garden-Home-Toy (SGHT). 

The preprocessed CDR datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1qFePm3zqAvW9WikUCsyC8VestVU6bvY5?usp=sharing). You can download them and place them in the `./data` path of this project.

## 4 Code Structure

```bash
FedHCDR
├── LICENSE                                     LICENSE file
├── README.md                                   README file 
├── checkpoint                                  Model checkpoints saving directory
│   └── ...
├── data                                        Data directory
│   └── ...
├── log                                         Log directory
│   └── ...
├── models                                      Local model packages
│   ├── __init__.py                             Package initialization file
│   ├── dhcf                                    dhcf package
│   │   ├── __init__.py                         Package initialization
│   │   ├── dhcf_model.py                       Model architecture
│   │   ├── config.py                           Model configuration file
│   │   └── modules.py                          Backbone modules (such as hyper GCN)
│   └── ...
├── pic                                         Picture directory
│   └── FedHCDR-Framework.png                   Model framework diagram
├──  utils                                      Tools such as data reading, IO functions, training strategies, etc.
│    ├── __init__.py                            Package initialization file
│    ├── data_utils.py                          Data reading (including ratings and graphs)
│    ├── io_utils.py                            IO functions
│    └── train_utils.py                         Training strategies
├── client.py                                   Client architecture   
├── dataloader.py                               Customized dataloader
├── dataset.py                                  Customized dataset          
├── fl.py                                       The overall process of federated learning
├── local_graph.py                              Local graph and hypergraph data structure
├── losses.py                                   Loss functions
├── main.py                                     Main function, including the complete data pipeline
├── requirements.txt                            Dependencies installation
├── server.py                                   Server-side model parameters and user representations aggregation
├── trainer.py                                  Training and test methods of FedHCDR and other baselines
└── .gitignore                                  .gitignore file
```


## 5 Train & Eval

### 5.1 Our method

To train FedHCDR (ours), you can run the following command:

```bash
python -u main.py \
        --num_round 60 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 1024 \
        --log_dir log \
        --method FedHCDR \
        --lr 0.001 \
        --seed 42 \
        --lam 2.0 \
        --gamma 2.0 \
        Food Kitchen Clothing Beauty
```
There are a few points to note:

- the positional arguments `Food Kitchen Clothing Beauty` indicates training FedHCDR in FKCB scenario. If you want to choose another scenario, you can change it to `Sports Clothing Elec Cell` (SCEC) or `Sports Garden Home Toys` (SGHT).

- The argument `--lam` is used to control local-global bi-directional knowledge transfer for FedHCDR method (ours). For FKCB, `2.0` is the best; for SCEC, `3.0` is the best; For SGHT, `1.0` is the best.

- The argument `--gamma` is used to control the intensity of hypergraph contrastive learning for FedHCDR method (ours). For FKCB, `2.0` is the best; for SCEC, `1.0` is the best; For SGHT, `3.0` is the best.

- If you restart training the model in a certain scenario, you can add the parameter `--load_prep` to load the dataset preprocessed (including ratings and graphs) in the previous training to avoid repeated data preprocessing.

To test FedHCDR, you can run the following command:
```bash
python -u main.py \
        --log_dir log \
        --method FedHCDR \
        --load_prep \
        --model_id 1709476223 \
        --do_eval \
        --seed 42 \
        Food Kitchen Clothing Beauty
```
Here `--model_id` is the model ID under which you saved the model before. You can check the ID of the saved models in the `checkpoint/domain_{$dataset}` directory.
### 5.2 Baselines

To train other baselines (LocalMF, LocalGNN, LocalDHCF, FedMF, FedGNN, PriCDR, FedP2FCDR, FedPPDM), you can run the following command:
```bash
python -u main.py \
        --num_round 60 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 1024 \
        --log_dir log \
        --method FedPPDM \
        --lr 0.001 \
        --seed 42 \
        Food Kitchen Clothing Beauty 
```
Here `FedPPDM` can be replaced with the name of the baselines you want to train.

For the local version without federated aggregation, you can run the following command:

```bash
python -u main.py \
        --num_round 60 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 1024 \
        --log_dir log \
        --method LocalPPDM \
        --lr 0.001 \
        --seed 42 \
        Food Kitchen Clothing Beauty 
```
Similarly, `FedPPDM` here can be replaced with the name of the baselines you want to train.


## 6 Citation
If you find this work useful for your research, please kindly cite FedHCDR by:
```text
@misc{zhang2024fedhcdr,
      title={FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling}, 
      author={Hongyu Zhang and Dongyi Zheng and Lin Zhong and Xu Yang and Jiyuan Feng and Yunqing Feng and Qing Liao},
      year={2024},
      eprint={2403.02630},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```