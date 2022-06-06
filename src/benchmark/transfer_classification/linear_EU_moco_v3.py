

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
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import builtins

from datasets.EuroSat.eurosat_dataset import EurosatDataset,Subset
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from models.moco_v3 import vits

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/resnet/')
parser.add_argument('--resume', type=str, default='')
#parser.add_argument('--save_path', type=str, default='./checkpoints/bigearthnet_s2_B12_100_no_pretrain_resnet50.pt')

parser.add_argument('--bands', type=str, default='B13', help='bands to process')  
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

parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--in_size',type=int,default=56)

parser.add_argument('--linear', action='store_true', default=False)

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
    

    data_dir = args.data_dir
    checkpoints_dir = args.checkpoints_dir
    #save_path = args.save_path
    batch_size = args.batchsize

    num_workers = args.num_workers
    epochs = args.epochs
    train_frac = args.train_frac
    seed = args.seed

    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir,exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))

    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(args.in_size),
            #cvtransforms.Resize(args.in_size),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor(),
            ])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(256),
            cvtransforms.CenterCrop(args.in_size),
            cvtransforms.ToTensor(),
            ])


    eurosat_dataset = EurosatDataset(root=args.data_dir,normalize=args.normalize)

    indices = np.arange(len(eurosat_dataset))
    train_indices, test_indices = train_test_split(indices, train_size=0.8,stratify=eurosat_dataset.targets,random_state=args.seed)    
  
    train_dataset = Subset(eurosat_dataset, train_indices, train_transforms)
    val_dataset = Subset(eurosat_dataset, test_indices, val_transforms)
        
        
    if train_frac is not None and train_frac<1:
        frac_indices = np.arange(len(train_dataset))
        sub_train_indices, sub_test_indices = train_test_split(frac_indices, train_size=train_frac, random_state=args.seed)
        train_dataset = Subset(train_dataset,sub_train_indices)    
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

    ###########################################################################
    
    net = vits.__dict__[args.backbone](in_chans=13, num_classes=10)
    linear_keyword = 'head'

    if args.linear:
        # freeze all layers but the last fc
        for name, param in net.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
        # init the fc layer
    getattr(net, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(net, linear_keyword).bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = net.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    
    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    if args.linear:
        assert len(parameters) == 2  # weight, bias

    #########################################################################


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9)


    last_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimzier.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_loss = checkpoint['loss']

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net.to(device)
    net.cuda()
    
    #### nccl doesn't support wsl
    if args.is_slurm_job:
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu_to_work_on],find_unused_parameters=True)

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
                score = torch.sigmoid(outputs).detach().cpu()            
                average_precision = accuracy_score(labels.cpu(), torch.argmax(score,axis=1)) * 100.0
            else:
                average_precision = 0

            score_time = time.time()-end-data_time-train_time
            
            # print statistics
            running_loss += loss.item()
            running_acc += average_precision
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

        if epoch%5==4:
            running_loss_val = 0.0
            running_acc_val = 0.0
            count_val = 0
            net.eval()
            with torch.no_grad():
                for j, data_val in enumerate(val_loader, 0):

                    inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()
                    outputs_val = net(inputs_val)
                    loss_val = criterion(outputs_val, labels_val.long())
                    score_val = torch.sigmoid(outputs_val).detach().cpu()
                    average_precision_val = accuracy_score(labels_val.cpu(), torch.argmax(score_val,axis=1)) * 100.0   

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
