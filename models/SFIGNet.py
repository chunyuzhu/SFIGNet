#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:54:52 2024

@author: z
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from models.KAN import KANLinear
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



    
    
class SFIGNet(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,hidden_dim=64,scale=4):
        super(SFIGNet, self).__init__()
        self.hsi_kan = KANLinear(HSI_bands,hidden_dim)
        self.msi_kan = KANLinear(MSI_bands,hidden_dim)
        self.HSI_bands = HSI_bands
        self.MSI_bands = MSI_bands
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.imnet_spa= KANLinear(hidden_dim*2+2,hidden_dim+1)
        self.imnet_fre_real= KANLinear(hidden_dim*2+2,hidden_dim+1)
        self.imnet_fre_imaginary= KANLinear(hidden_dim*2+2,hidden_dim+1)
        self.spec_adj = KANLinear(hidden_dim,HSI_bands)
    def query_spa(self, feat, coord, hr_guide):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]
        rx = 1 / h
        ry = 1 / w
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
               
                pred = self.imnet_spa(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]                
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret


    def query_fre_real(self, feat, coord, hr_guide):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]
        rx = 1 / h
        ry = 1 / w
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
               
                pred = self.imnet_fre_real(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret
    def query_fre_imaginary(self, feat, coord, hr_guide):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]
        rx = 1 / h
        ry = 1 / w
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                
                pred = self.imnet_fre_imaginary(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                # print('pred:',pred.shape)
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        # print('preds:',preds.shape)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret
    def forward(self, LRHSI, HRMSI):
        
        _, _, MSI_H, MSI_W = HRMSI.shape
        _, _, HSI_h, HSI_w = LRHSI.shape
        # shallow encoder
        lrhsi_feats = rearrange(LRHSI, 'b c h w -> b (h w) c')
        hrmsi_feats = rearrange(HRMSI, 'b c h w -> b (h w) c')
        lrhsi_feats = self.hsi_kan(lrhsi_feats)
        hrmsi_feats = self.msi_kan(hrmsi_feats)      
       
        lrhsi_feats = rearrange(lrhsi_feats, 'b (h w) c -> b c h w', h=HSI_h)        
        hrmsi_feats = rearrange(hrmsi_feats, 'b (h w) c -> b c h w', h=MSI_H)
        # print('hrmsi_feats:',hrmsi_feats.shape)
        # spatial guide
        B, c, H, W = hrmsi_feats.shape
        coord = make_coord([H, W]).cuda()        
        spa_guide = self.query_spa(lrhsi_feats, coord, hrmsi_feats) # BxCxHxW
        # print(';spa_guide:',spa_guide.shape)
        
        # frequency dec
        fft_hsi = torch.fft.fft2(lrhsi_feats)
        real_part_hsi = fft_hsi.real
        imaginary_part_hsi = fft_hsi.imag
        
        
        fft_msi = torch.fft.fft2(hrmsi_feats)     
        real_part_msi = fft_msi.real
        imaginary_part_msi = fft_msi.imag
        
        # real_part guide        
        real_part = self.query_fre_real(real_part_hsi, coord, real_part_msi)
        # print('real_part:',real_part.shape)
        # imaginary_part guide
        imaginary_part = self.query_fre_imaginary(imaginary_part_hsi, coord, imaginary_part_msi)
        # print('imaginary_part:',imaginary_part.shape)
        
        complex_tensor = torch.complex(real_part, imaginary_part)
        # print('complex_tensor:',complex_tensor.shape)
        
        ifre_guide = torch.fft.irfft2(complex_tensor,s=(H,W))
        # print('ifre_guide:',ifre_guide.shape)
        # print(ifre_guide)
        # print(spa_guide)
        # feature fusion
        feature = spa_guide + ifre_guide
        # print(feature.shape)
        feature = rearrange(feature, 'b c h w -> b (h w) c') 
        
        Out = self.spec_adj(feature)
        # print('aaaaaaaaaaaaaaaaaaaaaaa')
        Out = rearrange(Out, 'b (h w) c -> b c h w', h=MSI_H)
        return Out,0,0,0,0,0
            
if __name__ == '__main__':
    
    
        
    model = SFIGNet(HSI_bands=103,MSI_bands=4,hidden_dim=64,scale=8).cuda()
          
    HSI = torch.randn((1,103,16,16)).float().cuda()
    MSI = torch.randn((1,4,128,128)).float().cuda()
    
    T = model(HSI,MSI)
    print(T[0].shape)
        
        
        
            
            
            
            
