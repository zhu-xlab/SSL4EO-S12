import torch
import os
import argparse

parser = argparse.ArgumentParser(description='Convert pretrained model to torchvision or timm format')

parser.add_argument('--in_ckpt', help='input model path')
parser.add_argument('--out_ckpt', help='output model path')
parser.add_argument('--model', type=str, default='moco_v2_rn50')

args = parser.parse_args()

### MoCo-v2 + ResNet50
if os.path.isfile(args.in_ckpt):
    print("=> loading checkpoint '{}'".format(args.in_ckpt))
    checkpoint = torch.load(args.in_ckpt, map_location="cpu")
    
    
    
    if args.model=='moco_v2_rn50':
        state_dict = checkpoint['state_dict']    
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
            
        torch.save({'state_dict':state_dict},args.out_ckpt)
        
        print('Convert to:',args.out_ckpt)
        
    else:
        print('Error: unknown model.')
    