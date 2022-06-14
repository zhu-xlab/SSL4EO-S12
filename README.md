# SSL4EO-S12
The SSL4EO-S12 dataset is a large-scale mutilmodal multitemporal dataset for unsupervised/self-supervised pre-training in Earth observation. The dataset consists of unlabeled patch triplets (Sentinel-1 dual-pol SAR, Sentinel-2 top-of-atmosphere multispectral, Sentinel-2 surface reflectance multispectral) from 251079 locations across the globe, each patch covering 2640mx2640m and including four seasonal time stamps.

### Access to the dataset
The SSL4EO-S12 dataset is openly accessible at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Pre-trained models
The pre-trained models with different SSL methods are provided as follows (13 bands of S2-L1C, 100 epochs).


| SSL method |   Arch   | BigEarthNet | EuroSAT | So2Sat-LCZ42 |                                                   Download                                                  |          |      |
|:----------:|:--------:|:-----------:|:-------:|:------------:|:-----------------------------------------------------------------------------------------------------------:|:--------:|:----:|
|    MoCo    | ResNet50 |    91.8%    |  99.1%  |     60.9%    |    [full ckpt](https://syncandshare.lrz.de/getlink/fiUTyFN9kvFVhBfFBry6K8wK/B13_rn50_moco_0099_ckpt.pth)    | [backbone](https://syncandshare.lrz.de/getlink/fiYDLhcuBnssotKvLCwUTjad/B13_rn50_moco_0099.pth) | [logs](https://drive.google.com/file/d/1G66pdvJmeD6Rc-OZdOKA1h2Vnvq_0nnt/view?usp=sharing) |
|            | ViT-S/16 |    89.9%    |  98.6%  |     61.6%    |   [full ckpt](https://syncandshare.lrz.de/getlink/fiMJnvN2F2bi7enxbDduN9Tq/B13_vits16_moco_0099_ckpt.pth)   | [backbone](https://syncandshare.lrz.de/getlink/fi9nXfDQCgM37sBShifgZvDG/B13_vits16_moco_0099.pth) | [logs](https://drive.google.com/file/d/1f05B85T4Y2-RntfAw42uICKm9mwilHXF/view?usp=sharing) |
|    DINO    | ResNet50 |    90.7%    |  99.1%  |     63.6%    |    [full ckpt](https://syncandshare.lrz.de/getlink/fiEqiTz7JM2TFFxBa8D91mfo/B13_rn50_dino_0095_ckpt.pth)    | [backbone](https://syncandshare.lrz.de/getlink/fiUYZu2N7oNsfHHu6skVHJ3b/B13_rn50_dino_0099.pth) | logs |
|            | ViT-S/16 |    90.5%    |  99.0%  |     62.2%    |   [full ckpt](https://syncandshare.lrz.de/getlink/fi9mUJArfqKWtYeadixqxGfE/B13_vits16_dino_0099_ckpt.pth)   | [backbone](https://syncandshare.lrz.de/getlink/fiXrWtDAdYgEwbWGUBhcjxxc/B13_vits16_dino_0099.pth) | [logs](https://drive.google.com/file/d/1eeKrKFMa6akGyXugBRF6-rJ7oTIeZAno/view?usp=sharing) |
|     MAE    | ViT-S/16 |    88.9%    |  98.7%  |     63.9%    |    [full ckpt](https://syncandshare.lrz.de/getlink/fiXAvqk1spqizGLKaFpfENSX/B13_vits16_mae_0099_ckpt.pth)   | [backbone](https://syncandshare.lrz.de/getlink/fiKHsA3LyYLV8cUSMWphpUNE/B13_vits16_mae_0099.pth) | [logs](https://drive.google.com/file/d/1uJojq9q_fKMdD6cO1YXCPguZYEmfj35s/view?usp=sharing) |
|  Data2vec  | ViT-S/16 |    90.3%    |  99.1%  |     64.8%    | [full ckpt](https://syncandshare.lrz.de/getlink/fiV5t9MAya9UiV3U729ovyPN/B13_vits16_data2vec_0099_ckpt.pth) | [backbone](https://syncandshare.lrz.de/getlink/fi8GpGpL3zXmZ6ETPDTmeLGT/B13_vits16_data2vec_0099.pth) | logs |

Other pre-trained models:

| SSL method |   Arch   | Input |                                                           Download                                                           |          |      |
|:----------:|:--------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------------:|:--------:|:----:|
|    MoCo    | ResNet18 | S2-L1C 13 bands      |             [full ckpt](https://syncandshare.lrz.de/getlink/fiLnquyqkNtaqC8kbBiMuM7R/B13_rn18_moco_0099_ckpt.pth)            | backbone | logs |
|            | ResNet18 | S2-L1C RGB            | [full ckpt](https://syncandshare.lrz.de/getlink/fiParGkEcSEB3WJPYPc19rBP/B3_rn18_moco_0099_ckpt.pth), [full ckpt ep200](https://syncandshare.lrz.de/getlink/fiYGJWR9X9r63sLjdYzeMpRy/B3_rn18_moco_0199_ckpt.pth) | backbone | logs |
|            | ResNet50 | S2-L1C RGB            |             [full ckpt](https://syncandshare.lrz.de/getlink/fiYTwDhKBLM8zQr1gBqXdQvN/B3_rn50_moco_0099_ckpt.pth)            | backbone | logs |
|            | ResNet50 | S1 2 bands            |             [full ckpt](https://syncandshare.lrz.de/getlink/fiGkSX6Xfdm8dpnW2K9cJMdh/B2_rn50_moco_0099_ckpt.pth)            | backbone | logs |

