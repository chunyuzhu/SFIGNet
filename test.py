import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
import numpy as np
from metrics import *
from models.TSFN import TSFN
# # from models.MambaINR import MambaINR
from models.PSRTNet import PSRTnet
from models._3DT_Net import _3DT_Net
from models.SSRNET import SSRNET
from models.HyperKite import HyperKite
from models.Kalman_fusion import Kal_KAN
# from models.AMGSGAN import AMGSGAN
# from models.TFNet import ResTFNet
# from models.MoGDCNx4 import MoGDCNx4
# from models.MoGDCN import MoGDCN
# from models.MoGDCNx16 import MoGDCNx16
from models.KANFusion import KANFusion
# from models.KNLNet import KNLNet 
# from models.DCT import DCT
from models.ITF import ITF
from models.KANFusion_no_fre import KANFusion_no_fre
from models.KANFusion_no_spa import KANFusion_no_spa
from models.KANFusion_mlp import KANFusion_mlp
from models.KAN_F_G import KAN_F_G
# from models.P3Net import P3Net
# from models.P3Net_No_Point import P3Net_No_Point
# from models.P3Net_No_Patch import P3Net_No_Patch
# from models.P3Net_No_Panoramic import P3Net_No_Panoramic
# from models.P3Net_No_Patch_Panoramic import P3Net_No_Patch_Panoramic
# from models.Fusformer import Fusformer
from data_loader import build_datasets
# from models.PPGMamba import PPGMamba
# from models.Mamba_enhance_INR import Mamba_INR
import pandas as pd
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F
import cv2
from time import *
import os
import scipy.io as io
from thop import profile

torch.cuda.is_available()

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

# torch.cuda.is_available()
def main():
    # if args.dataset == 'PaviaU':
    #   args.n_bands = 103
    # elif args.dataset == 'Botswana':
    #   args.n_bands = 145
    # elif args.dataset == 'Chikusei':
    #   args.n_bands = 128
    # elif args.dataset == 'IEEE2018':
    #   args.n_bands = 48
    

    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)

    # Build the models
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'Chikusei':
      args.n_bands = 128
    elif args.dataset == 'IEEE2018':
      args.n_bands = 48
    elif args.dataset == 'Houston':
      args.n_bands = 144

    # Build the models
    if args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'MGDIN':
      model = MGDIN(args.scale_ratio,
                    args.n_select_bands,
                    args.n_bands).cuda()
    elif args.arch == 'PSRTnet':
      model = PSRTnet(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands,
                     args.image_size).cuda()
    elif args.arch == 'Kal_KAN':
      model = Kal_KAN(HSI_bands=args.n_bands,
                              MSI_bands=args.n_select_bands,
                              hidden_dim=64,scale=args.scale_ratio,depth=4,image_size=64).cuda()
    elif args.arch == 'ResTFNet':
      model = ResTFNet(args.scale_ratio, 
                       args.n_select_bands,
                       args.n_bands,
                       ).cuda()
    elif args.arch == 'HyperKite':
      model = HyperKite(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'TSFN':
      model = TSFN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'MoGDCNx4':
      model = MoGDCNx4(scale_ratio=args.scale_ratio,
                       n_select_bands=args.n_select_bands, 
                       n_bands=args.n_bands,
                       img_size=args.image_size).cuda()
    elif args.arch == 'MoGDCN':
      model = MoGDCN(scale_ratio=args.scale_ratio,
                       n_select_bands=args.n_select_bands, 
                       n_bands=args.n_bands,
                       img_size=args.image_size).cuda()
    elif args.arch == 'MoGDCNx16':
      model = MoGDCNx16(scale_ratio=args.scale_ratio,
                       n_select_bands=args.n_select_bands, 
                       n_bands=args.n_bands,
                       img_size=args.image_size).cuda()
    elif args.arch == '_3DT_Net':
      model = _3DT_Net(args.scale_ratio, 8, 
                       args.n_bands,args.n_select_bands
                       ).cuda()
    elif args.arch == 'KANFusion':
      model = KANFusion(args.n_bands,
                        args.n_select_bands,
                        64,
                        args.scale_ratio,
                        depth=4,
                        image_size=128
                       ).cuda()
    elif args.arch == 'ITF':
      model = ITF(args.scale_ratio, 
                    args.n_select_bands, 
                    args.n_bands,
                    feat_dim = 64,
                    local_ensemble=True, 
                   feat_unfold=False, 
                   scale_token=True).cuda()
    elif args.arch == 'KAN_F_G':
      model = KAN_F_G(args.n_bands,
                        args.n_select_bands,
                        64,
                        args.scale_ratio,
                        depth=4,
                        image_size=128
                       ).cuda()
    elif args.arch == 'KANFusion_no_fre':
       model = KANFusion_no_fre(args.n_bands,
                         args.n_select_bands,
                         64,
                         args.scale_ratio,
                         depth=4,
                         image_size=128
                        ).cuda()    
    elif args.arch == 'KANFusion_no_spa':
       model = KANFusion_no_spa(args.n_bands,
                         args.n_select_bands,
                         64,
                         args.scale_ratio,
                         depth=4,
                         image_size=128
                        ).cuda()    
    elif args.arch == 'KANFusion_mlp':
      model = KANFusion_mlp(args.n_bands,
                        args.n_select_bands,
                        80,
                        args.scale_ratio,
                        depth=4,
                        image_size=128
                       ).cuda()
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()
    
    begin_time = time()
    if args.arch == 'SSRNET':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpatRNET':
        _, out, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpecRNET':
        _, _, out, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SwinCGAN':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda(), args.scale_ratio)
    else:
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    end_time = time()
    run_time = (end_time-begin_time)*1000

    print ()
    print ()
    print ('Dataset:   {}'.format(args.dataset))
    print ('Arch:   {}'.format(args.arch))
    print ('ModelSize(M):   {}'.format(np.around(os.path.getsize(model_path)//1024/1024.0, decimals=2)))
    print ('Time(Ms):   {}'.format(np.around(run_time, decimals=2)))
    flops, params = profile(model, inputs=(lr.cuda(),hr.cuda()))
    flops = flops/1000000000
    print ('flops:',flops)
    print('params_bm:',sum([param.nelement() for param in model.parameters()])/ 1e6)
    print ('params:',params/1000000)
    
    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    
    slr  =  F.interpolate(lr, scale_factor=args.scale_ratio, mode='bilinear')
    slr = slr.detach().cpu().numpy()
    slr  =  np.squeeze(slr).transpose(1,2,0).astype(np.float64)
    
    sref = np.squeeze(ref).transpose(1,2,0).astype(np.float64)
    sout = np.squeeze(out).transpose(1,2,0).astype(np.float64)
    
    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+args.arch+'.mat',{'Out':sout})
    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'REF.mat',{'REF':sref})
    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'Upsample.mat',{'Out':slr})
    
    t_lr = np.squeeze(lr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    t_hr = np.squeeze(hr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    
    io.savemat('./为传统方法准备数据/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'lr'+'.mat',{'HSI':t_lr})
    io.savemat('./为传统方法准备数据/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'hr'+'.mat',{'MSI':t_hr})
    
    
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    print ('RMSE:   {:.4f};'.format(rmse))
    print ('PSNR:   {:.4f};'.format(psnr))
    print ('ERGAS:   {:.4f};'.format(ergas))
    print ('SAM:   {:.4f}.'.format(sam))

   
if __name__ == '__main__':
    main()
