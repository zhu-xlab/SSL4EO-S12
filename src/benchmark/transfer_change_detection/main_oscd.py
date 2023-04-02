from pathlib import Path
# from copy import deepcopy
from argparse import ArgumentParser

import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.models.resnet import resnet18
from torchvision.models import resnet50
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Precision, Recall, F1Score

from datasets.oscd_datamodule import ChangeDetectionDataModule
from models.segmentation import get_segmentation_model
# from models.moco2_module import MocoV2
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--resnet_type', type=int, default=18)
    parser.add_argument('--init_type', type=str, default='random')
    parser.add_argument('--ckp_path', type=str, default=None)
    parser.add_argument('--n_channels', dest='nc', type=int, default=3)
    parser.add_argument('--n_epochs', dest='ne', type=int, default=100)
    parser.add_argument('--learning_rate', dest='lr', type=float, default=0.001)
    parser.add_argument('--value_discard', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--m_threshold', dest='mth', type=float, default=0.5)
    parser.add_argument('--result_dir', type=str)
    
    args = parser.parse_args()
    
    return args


def dice_loss(out,mask,epsilon=1e-5):
    inter = torch.dot(out.reshape(-1), mask.reshape(-1))
    sets_sum = torch.sum(out) + torch.sum(mask)
    return (2 * inter + epsilon) / (sets_sum + epsilon)


class SiamSegment(LightningModule):

    def __init__(self, backbone, feature_indices, feature_channels):
        super().__init__()
        self.model = get_segmentation_model(backbone, feature_indices, feature_channels)
        self.criterion = BCEWithLogitsLoss()
        self.dice_loss = dice_loss
        self.prec = Precision(task='binary',threshold=args.mth)
        self.rec = Recall(task='binary',threshold=args.mth) 
        self.f1 = F1Score(task='binary',threshold=args.mth) 

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True,sync_dist=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        #if args.nc == 3:
        #    tensorboard.add_image('train/img_1', img_1[0], global_step)
        #    tensorboard.add_image('train/img_2', img_2[0], global_step)
        #    tensorboard.add_image('train/mask', mask[0], global_step)
        #    tensorboard.add_image('train/out', pred[0], global_step)
        #else:
        #    tensorboard.add_image('train/img_1', img_1[0,1:4,:,:], global_step)
        #    tensorboard.add_image('train/img_2', img_2[0,1:4,:,:], global_step)
        #    tensorboard.add_image('train/mask', mask[0], global_step)
        #    tensorboard.add_image('train/out', pred[0], global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True,sync_dist=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        #if args.nc == 3:
        #    tensorboard.add_image('val/img_1', img_1[0], global_step)
        #    tensorboard.add_image('val/img_2', img_2[0], global_step)
        #    tensorboard.add_image('val/mask', mask[0], global_step)
        #    tensorboard.add_image('val/out', pred[0], global_step)
        #else:
        #    tensorboard.add_image('val/img_1', img_1[0,1:4,:,:], global_step)
        #    tensorboard.add_image('val/img_2', img_2[0,1:4,:,:], global_step)
        #    tensorboard.add_image('val/mask', mask[0], global_step)
        #    tensorboard.add_image('val/out', pred[0], global_step)
        return loss

    def shared_step(self, batch):
        img_1, img_2, mask = batch
        out = self(img_1, img_2)
        pred = torch.sigmoid(out)
        loss = self.criterion(out, mask)
        # + self.dice_loss(out,mask)
        prec = self.prec(pred, mask.long())
        rec = self.rec(pred, mask.long())
        f1 = self.f1(pred, mask.long())
        return img_1, img_2, mask, pred, loss, prec, rec, f1

    def configure_optimizers(self):
        # params = self.model.parameters()
        params = set(self.model.parameters()).difference(self.model.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


def load_ssl_resnet_encoder(net, ckp_path, pf_sdict='module.encoder_q.'):
    # load ckp file
    state_dict = torch.load(ckp_path, map_location='cpu')['state_dict']

    # modify keys in state_dict
    for k in list(state_dict.keys()):
        if k.startswith(pf_sdict) and not k.startswith(pf_sdict+'fc'):
            newk = k[len(pf_sdict):]
            state_dict[newk] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    #load weights
    msg = net.load_state_dict(state_dict,strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    
    # print ckp info after succussfully loading it
    print(f"Checkpoint has been loaded from \'{ckp_path}\'!")
    
    return net


if __name__ == '__main__':
    pl.seed_everything(42)
    
    # args
    args = get_args()

    # dataloader
    assert(args.nc==3 or args.nc==13 or args.nc==12)
    datamodule = ChangeDetectionDataModule(args.data_dir, RGB_bands=True if args.nc==3 else False, \
                                           BGR_bands=False, \
                                           S2A_bands=True if args.nc==12 else False, \
                                           value_discard=args.value_discard, \
                                           patch_size=args.patch_size, batch_size=args.batch_size)

    # construct backbone model
    pretrained = False
    # check whether use imagenet pretrained weights, which are only applied for RGB model
    if args.init_type == 'imagenet':
        assert(args.nc==3)
        pretrained = True
    # backbone definition
    if args.resnet_type == 18:
        backbone = resnet18(pretrained=pretrained)
        feature_channels = (64, 64, 128, 256, 512)
    elif args.resnet_type == 50:
        backbone = resnet50(pretrained=pretrained)
        feature_channels=(64, 256, 512, 1024, 2048)
    else:
        raise ValueError()
    print(f'Construct the backbone of resnet{args.resnet_type}-initialization: {args.init_type}.')
    
    # change the number of input channels of backbone
    if args.nc != 3:
        backbone.conv1 = torch.nn.Conv2d(args.nc, 64, 7, stride=2, padding=3, bias=False)
        print(f'Modify the number of inputs of the backbone to {args.nc}.')
    
    # fix backbone layers
    for name, param in backbone.named_parameters():
            param.requires_grad = False 
    
    # load ckp if given
    if args.ckp_path:
        backbone = load_ssl_resnet_encoder(backbone, args.ckp_path)
        # args.init_type = 'ssl'

    model = SiamSegment(backbone, feature_indices=(2, 4, 5, 6, 7), feature_channels=feature_channels)
    # model.example_input_array = (torch.zeros((1, 3, 96, 96)), torch.zeros((1, 3, 96, 96)))

    experiment_name = args.init_type
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / args.result_dir / 'logs'), name=experiment_name)
    dir_path=str(Path.cwd() / args.result_dir / 'ckps' / args.init_type)
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=dir_path, auto_insert_metric_name=False, filename='{epoch}-{val/loss:.2f}-{val/f1:.2f}', save_weights_only=False)
    early_stop_callback = EarlyStopping(monitor="val/loss", mode="min")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval=None)
    
    trainer = Trainer(accelerator='gpu', devices=4, strategy="ddp", sync_batchnorm=True, logger=logger, callbacks=[checkpoint_callback,lr_monitor], max_epochs=args.ne, enable_progress_bar=True)
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
