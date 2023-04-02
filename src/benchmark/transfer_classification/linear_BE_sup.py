"""
supervised training of BigEarthNet (all bands) with resnet18/50

TODO:
  -- optimize and reduce RAM usage
  -- optimize I/O
  -- checkpoints
  -- merge B12 and RGB codes

"""



import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models

## change01 ##
from cvtorchvision import cvtransforms
import time
import os
import math
import pdb
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_f1_score
import numpy as np
import argparse
import builtins

from datasets.BigEarthNet.bigearthnet_dataset_seco import Bigearthnet
from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb_s2_uint8 import LMDBDataset,random_subset

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet')
parser.add_argument('--lmdb_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/resnet/')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--save_path', type=str, default='./checkpoints/bigearthnet_s2_B12_100_no_pretrain_resnet50.pt')

parser.add_argument('--bands', type=str, default='all', choices=['all','RGB','B12'], help='bands to process')  
parser.add_argument('--train_frac', type=float, default=1.0)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')

### distributed running ###
parser.add_argument('--dist_url', default='env://', type=str)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

parser.add_argument('--normalize',action='store_true',default=False)
parser.add_argument('--linear',action='store_true',default=False)
parser.add_argument('--pretrain_style',default=None,type=str,choices=['reinit','pad',None])


def init_distributed_mode(args):

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])


    # prepare distributed
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return    

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    global args
    args = parser.parse_args()
    ### dist ###
    init_distributed_mode(args)
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    fix_random_seeds(args.seed)
    
    # `python bigearthnet_dataset.py` to create lmdb data
    lmdb = True # use lmdb dataset
    data_dir = args.data_dir
    lmdb_dir = args.lmdb_dir
    checkpoints_dir = args.checkpoints_dir
    save_path = args.save_path
    batch_size = args.batchsize

    num_workers = args.num_workers
    epochs = args.epochs
    train_frac = args.train_frac
    seed = args.seed

    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))


    ## change02 ##
    if args.bands == 'RGB':
        bands = ['B04', 'B03', 'B02']
        lmdb_train = 'train_RGB.lmdb'
        lmdb_val = 'val_RGB.lmdb'
    else:
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        lmdb_train = 'train_B12.lmdb'
        lmdb_val = 'val_B12.lmdb'
        
    num_labels = 19

    ## change03 ##
    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224,scale=(0.8,1.0)), # multilabel, avoid cropping out labels
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor()])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(256),
            cvtransforms.CenterCrop(224),
            cvtransforms.ToTensor(),
            ])

    if lmdb:
        train_dataset = LMDBDataset(
            lmdb_file=os.path.join(lmdb_dir, lmdb_train),
            transform=train_transforms
        )
        
        val_dataset = LMDBDataset(
            lmdb_file=os.path.join(lmdb_dir, lmdb_val),
            transform=val_transforms
        )    
    else:
        train_dataset = Bigearthnet(
            root=data_dir,
            split='train',
            bands=bands,
            use_new_labels = True,
            transform=train_transforms
        )
        
        val_dataset = Bigearthnet(
            root=data_dir,
            split='val',
            bands=bands,
            use_new_labels = True,
            transform=train_transforms
        )    
        
        
    if train_frac is not None and train_frac<1:
        train_dataset = random_subset(train_dataset,train_frac,seed)    
    ### dist ###    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)    
        
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler = sampler,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True
                              
                              )
                              
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True
                              
                              )
    
    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))

    ## change 04 ##
    if args.backbone == 'resnet50':
        net = models.resnet50(pretrained=False)
    elif args.backbone == 'resnet18':
        net = models.resnet18(pretrained=False)
        
    if args.pretrain_style == 'reinit':
        print("=> loading checkpoint '{}'".format(args.pretrained))
        state_dict = torch.load(args.pretrained, map_location="cpu")
        msg = net.load_state_dict(state_dict, strict=False)
        #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(args.pretrained))        
         
    if args.bands=='all':
        net.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif args.bands=='B12':
        net.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrain_style == 'pad':
        print("=> loading checkpoint '{}'".format(args.pretrained))
        state_dict = torch.load(args.pretrained, map_location="cpu")
        new_conv1_weight = torch.zeros((64,12,7,7)) # 64,12,7,7
        # B10 init with 0
        new_conv1_weight[:,1,:,:] = state_dict['conv1.weight'][:,2,:,:]
        new_conv1_weight[:,2,:,:] = state_dict['conv1.weight'][:,1,:,:]
        new_conv1_weight[:,3,:,:] = state_dict['conv1.weight'][:,0,:,:]
        
        state_dict['conv1.weight'] = new_conv1_weight

        msg = net.load_state_dict(state_dict, strict=False)
        #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(args.pretrained))


    if args.backbone == 'resnet50':
        net.fc = torch.nn.Linear(2048,19)
    elif args.backbone == 'resnet18':
        net.fc = torch.nn.Linear(512,19)   

    if args.linear:
        for name, param in net.named_parameters():
            if name not in ['fc.weight','fc.bias']:
                param.requires_grad = False

        net.fc.weight.data.normal_(mean=0.0,std=0.01)
        net.fc.bias.data.zero_()

    # convert batch norm layers (if any)
    if args.is_slurm_job:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    net.cuda()
    #### nccl doesn't support wsl
    if args.is_slurm_job:
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu_to_work_on],find_unused_parameters=True)
        
        
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)


    last_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['model_state_dict']
        #state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
        net.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_loss = checkpoint['loss']

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net.to(device)
    #net.cuda()
    


    print('Start training...')
    for epoch in range(last_epoch,epochs):

        net.train()
        adjust_learning_rate(optimizer, epoch, args)
        
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        running_acc = 0.0
        
        running_loss_epoch = 0.0
        running_acc_epoch = 0.0
        
        start_time = time.time()
        end = time.time()
        sum_bt = 0.0
        sum_dt = 0.0
        sum_tt = 0.0
        sum_st = 0.0
        for i, data in enumerate(train_loader, 0):
            data_time = time.time()-end
            #inputs, labels = data
            if args.bands=='all':
                b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                images = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:,:,:]),dim=1)            
                inputs, labels = images.cuda(), data[1].cuda()
            else:    
                inputs, labels = data[0].cuda(), data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #pdb.set_trace()
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_time = time.time()-end-data_time
            if epoch%5==4:
                score = torch.sigmoid(outputs).detach()
                #average_precision = average_precision_score(labels.cpu(), score, average='micro') * 100.0
                average_precision = multilabel_average_precision(score, labels, num_labels=19, average="micro") * 100.0
            else:
                average_precision = 0
            score_time = time.time()-end-data_time-train_time
            
            # print statistics
            running_loss += loss.item()
            #running_acc += average_precision
            batch_time = time.time() - end
            end = time.time()        
            sum_bt += batch_time
            sum_dt += data_time
            sum_tt += train_time
            sum_st += score_time
            
            if i % 20 == 19:    # print every 20 mini-batches

                print('[%d, %5d] loss: %.3f acc: %.3f batch_time: %.3f data_time: %.3f train_time: %.3f score_time: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20, running_acc / 20, sum_bt/20, sum_dt/20, sum_tt/20, sum_st/20))
                
                #train_iter =  i*args.batch_size / len(train_dataset)
                #tb_writer.add_scalar('train_loss', running_loss/20, global_step=(epoch+1+train_iter) )
                running_loss_epoch = running_loss/20
                running_acc_epoch = running_acc/20
                
                running_loss = 0.0
                running_acc = 0.0
                sum_bt = 0.0
                sum_dt = 0.0
                sum_tt = 0.0
                sum_st = 0.0

        if epoch % 5 == 4:
            running_loss_val = 0.0
            running_acc_val = 0.0
            count_val = 0
            net.eval()
            with torch.no_grad():
                for j, data_val in enumerate(val_loader, 0):

                    if args.bands=='all':
                        b_zeros = torch.zeros((data_val[0].shape[0],1,data_val[0].shape[2],data_val[0].shape[3]),dtype=torch.float32)
                        images = torch.cat((data_val[0][:,:10,:,:],b_zeros,data_val[0][:,10:,:,:]),dim=1)
                        inputs_val, labels_val = images.cuda(), data_val[1].cuda()
                    else:
                        inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()

                    outputs_val = net(inputs_val)
                    loss_val = criterion(outputs_val, labels_val.long())
                    score_val = torch.sigmoid(outputs_val).detach()
                    #average_precision_val = average_precision_score(labels_val.cpu(), score_val, average='micro') * 100.0
                    average_precision_val = multilabel_average_precision(score_val, labels_val, num_labels=19, average="micro") * 100.0
                       

                    count_val += 1
                    running_loss_val += loss_val.item()
                    running_acc_val += average_precision_val        

            print('Epoch %d val_loss: %.3f val_acc: %.3f time: %s seconds.' % (epoch+1, running_loss_val/count_val, running_acc_val/count_val, time.time()-start_time))

            if args.rank == 0:
                losses = {'train': running_loss_epoch,
                          'val': running_loss_val/count_val}
                accs = {'train': running_acc_epoch,
                        'val': running_acc_val/count_val}        
                tb_writer.add_scalars('loss', losses, global_step=epoch+1, walltime=None)
                tb_writer.add_scalars('acc', accs, global_step=epoch+1, walltime=None)
        
            
            
        if args.rank==0 and epoch % 10 == 9:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':loss,
                        }, os.path.join(checkpoints_dir,'checkpoint_{:04d}.pth.tar'.format(epoch)))
        
    #if args.rank==0:
    #    torch.save(net.state_dict(), save_path)
        
    print('Training finished.')



if __name__ == "__main__":
    main()
