# SSL4EO-S12
SSL4EO-S12: a large-scale mutilmodal multitemporal dataset for self-supervised learning in Earth observation

### News
- [Jun14, 2022] The dataset has been published at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Dataset access
The SSL4EO-S12 dataset is openly accessible at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Pre-trained models

| SSL method |   Arch   | BigEarthNet | EuroSAT | So2Sat-LCZ42 |                                                Download                                               |          |      |
|:----------:|:--------:|:-----------:|:-------:|:------------:|:-----------------------------------------------------------------------------------------------------:|:--------:|:----:|
|    MoCo    | ResNet50 |             |         |              | [full ckpt](https://syncandshare.lrz.de/getlink/fiUTyFN9kvFVhBfFBry6K8wK/B13_rn50_moco_0099_ckpt.pth) | backbone | logs |
|            | ViT-S/16 |             |         |              |                                               full ckpt                                               | backbone | logs |
|    DINO    | ResNet50 |             |         |              |                                               full ckpt                                               | backbone | logs |
|            | ViT-S/16 |             |         |              |                                               full ckpt                                               | backbone | logs |
|     MAE    | ViT-S/16 |             |         |              |                                               full ckpt                                               | backbone | logs |
|  Data2vec  | ViT-S/16 |             |         |              |                                               full ckpt                                               | backbone | logs |


TODOs:
- [ ] add pre-trained weights
- [ ] organize the codes


