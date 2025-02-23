import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
# # from models.MambaINR import MambaINR
# from models.PSRTNet import PSRTnet


from models.SFIGNet import SFIGNet

from data_loader import build_datasets
# from models.PPGMamba import PPGMamba
# from models.Mamba_enhance_INR import Mamba_INR
import pandas as pd
from utils import *
# from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F


args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

df = pd.read_excel(args.dataset+'_rand.xlsx', sheet_name='Sheet1')
h_rand = df.iloc[:,0].tolist()
w_rand = df.iloc[:,1].tolist()
def main():
    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Chikusei':
      args.n_bands = 128
    elif args.dataset == 'IEEE2018':
      args.n_bands = 48
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'Houston':
      args.n_bands = 144
    # Build the models
    if args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'DCT':
      model = DCT(n_colors=args.n_bands, upscale_factor=args.scale_ratio, n_feats=180).cuda()
    elif args.arch == 'PSRTnet':
      model = PSRTnet(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands,
                     args.image_size).cuda()
    
    
    elif args.arch == 'SFIGNet':
      model = SFIGNet(args.n_bands,
                        args.n_select_bands,
                        64,
                        args.scale_ratio
                       ).cuda()
    

   
        
    # Loss and optimizer
    # criterion = nn.MSELoss().cuda()
    criterion = nn.L1Loss().cuda()
    #criterion = MAE_SAM_LOGloss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                0,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

    best_psnr = 0
    best_psnr = validate(test_list,
                          args.arch, 
                          model,
                          0,
                          args.n_epochs)
    print ('psnr: ', best_psnr) 

    # Epochs
    print ('Start Training: ')
    for epoch in range(args.n_epochs):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        h_str = h_rand[epoch]
        w_str = w_rand[epoch]
        train(train_list, 
              args.image_size,
              args.scale_ratio,
              args.n_bands, 
              args.arch,
              model, 
              optimizer, 
              criterion, 
              epoch, 
              args.n_epochs,
              h_str ,
              w_str)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)
        print ('best_psnr: ', best_psnr)
        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
          torch.save(model.state_dict(), model_path)
          print ('Saved!')
          print ('')

    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    main()
