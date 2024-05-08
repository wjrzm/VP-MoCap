import json
import cv2
import os
import os.path as osp
import numpy as np
import torch
import random
import pickle, glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from icecream import ic

from lib.Dataset.ImageModule import ImageModule
from lib.Dataset.InsoleModule import InsoleModule


class ContDataset(Dataset):
    def __init__(self, opt, phase):
        self.phase = phase
        self.basedir = opt.datadir
        self.seq_name = opt.seq_name

        self.insole_module = InsoleModule(self.basedir)
        self.image_module = ImageModule(opt)
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        self.w_sc = opt.w_sc
        self.w_nc = opt.w_nc

        self.img_res = opt.img_res
        self.is_aug = opt.aug.is_aug
        self.scale_factor = opt.aug.scale_factor  # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = opt.aug.noise_factor
        self.rot_factor = opt.aug.rot_factor  # Random rotation in the range [-rot_factor, rot_factor]

        self.is_sam = opt.aug.is_sam
        self.img_W = 1280
        self.img_H = 720

        self.sub_info = {}
        self.sub_info['20230422'] = np.load(osp.join(self.basedir, '20230422/sub_info.npy'), allow_pickle=True).item()
        self.sub_info['20230611'] = np.load(osp.join(self.basedir, '20230611/sub_info.npy'), allow_pickle=True).item()
        self.sub_info['20230713'] = np.load(osp.join(self.basedir, '20230713/sub_info.npy'), allow_pickle=True).item()

        self.seqlen = int(opt.tv_fn[-5:-4])
        self.halfseqlen = int((self.seqlen - 1) / 2)

        self.images_fn = []
        train_val = np.load(opt.tv_fn, allow_pickle=True).item()
        data_ls = train_val[self.phase]

        for data_id in data_ls.keys():
            sub_ls = data_ls[data_id]
            for sub_id in sub_ls.keys():
                seq_ls = sub_ls[sub_id]
                for seq_name in seq_ls.keys():
                    insole_ls = seq_ls[seq_name]
                    images_fn = sorted([os.path.join(data_id, sub_id, seq_name, x[:-4])
                                        for x in insole_ls])
                    self.images_fn += images_fn

        if self.phase == 'train':
            random.shuffle(self.images_fn)

        self.insole_mask = torch.from_numpy(self.insole_module.maskImg)

    def augm_params(self, flip, sc):
        """Get augmentation parameters."""
        # We flip with probability 1/2
        if np.random.uniform() <= 0.5:
            flip = 1

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1 + self.scale_factor,
                 max(1 - self.scale_factor, np.random.randn() * self.scale_factor + 1))
        if sc > 1:
            sc = 2 - sc
        # but it is zero with probability 3/5
        # rebg = np.random.uniform()
        return flip, sc

    def __len__(self):
        return len(self.images_fn)

    def __getitem__(self, index):
        case_name = self.images_fn[index]
        # case_name = 'S01/MoCap_20230422_092324/000' #debug
        data_id, sub_ids, seq_name, frame_ids = case_name.split('/')
        frame_idx = int(frame_ids)
        seq_dir = osp.join(self.basedir, data_id, sub_ids, seq_name)

        # load insole
        sub_weight = self.sub_info[data_id][sub_ids]['weight']
        insole_data = np.load(osp.join(seq_dir, 'insole/%s.npy' % frame_ids), allow_pickle=True).item()
        insole_press = insole_data['insole']

        insole = self.insole_module.sigmoidNorm(insole_press, sub_weight)
        # cv2.imwrite('debug/tmp.png',self.insole_module.showNormalizedInsole(insole))
        insole = np.concatenate([insole[0], insole[1]], axis=1)
        insole = torch.from_numpy(insole).float()

        # calc contact
        contact_map = self.insole_module.press2Cont(insole_press, sub_weight, th=0.5)
        # cv2.imwrite('debug/cont.png',self.insole_module.showContact(contact_map))

        # Aug
        flip = 0  # flipping
        sc = 1  # scaling
        if self.is_aug and self.phase == 'train':
            flip, sc = self.augm_params(flip, sc)

        kp_ls = []
        for frame_i in range(-self.halfseqlen, self.halfseqlen + 1):
            kp_fn = osp.join(seq_dir, 'keypoints/%03d.npy' % (frame_idx + frame_i))
            kp_data = np.load(kp_fn, allow_pickle=True).item()
            keypoints = np.array(kp_data['keypoints'])  # [width,height]
            keypoints[:, 0] = keypoints[:, 0]  # /self.img_W
            keypoints[:, 1] = keypoints[:, 1]  # /self.img_H
            keypoint_scores = np.array(kp_data['keypoint_scores']).reshape([-1, 1])
            kps = np.concatenate([keypoints, keypoint_scores], axis=1)
            kps = torch.from_numpy(kps).float()
            kp_ls.append(kps)

        # normalized keypoints
        kp_temp = torch.stack(kp_ls)
        kp_h = torch.max(kp_temp[..., 0]) - torch.min(kp_temp[..., 0])
        kp_w = torch.max(kp_temp[..., 1]) - torch.min(kp_temp[..., 1])
        kp_center = kp_temp[2, 19, :2]  # root keypoints
        kp_temp[:, :, 0] = kp_temp[:, :, 0] - kp_center[0]
        kp_temp[:, :, 1] = kp_temp[:, :, 1] - kp_center[1]
        kp_temp[..., 0] = kp_temp[..., 0] / kp_h
        kp_temp[..., 1] = kp_temp[..., 1] / kp_w

        if flip and self.phase == 'train':
            # if True:
            kp_temp[:, :, 0] = -kp_temp[:, :, 0]
            insole = torch.flip(insole, dims=[1])
            # contact_map = torch.flip(torch.from_numpy(contact_map).float(), dims=[1])
            contact_map = np.flip(contact_map, axis=1)
            contact_map = contact_map.copy()

        _smpl_cont = self.insole_module.getVertsPress(np.stack([contact_map[:, :11], contact_map[:, 11:]]))
        smpl_cont = np.concatenate([_smpl_cont[0], _smpl_cont[1]])

        contact_map = torch.from_numpy(contact_map).float()
        smpl_cont = torch.from_numpy(smpl_cont).float()

        if self.phase == 'train':
            kp_temp[:, :, :2] = kp_temp[:, :, :2] * sc

        res = {
            'case_name': case_name,
            'keypoints': kp_temp,
            'insole': insole[self.insole_mask],
            'contact_label': contact_map[self.insole_mask],
            'contact_smpl': smpl_cont,
        }
        return res
