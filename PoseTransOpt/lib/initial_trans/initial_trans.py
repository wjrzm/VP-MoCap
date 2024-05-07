import os
import os.path as osp
import numpy as np
import torch

from icecream import ic

from models.smpl import SMPL
from lib.initial_trans.transpose.model import ParametricModel
from lib.initial_trans.transolver import TranSolver
from lib.utils.depth_utils import depth_to_pointcloud
from lib.utils.contact_utils import find_contact_change_idx

def initial_trans(cfg, dataset):

    init_trans_path = cfg['task']['input_path_base']
    if not os.path.isdir(init_trans_path):
        os.makedirs(init_trans_path)
    
    transolver = TranSolver(cfg, 'models/SMPL_MALE.pkl', floor=0.0, scale=cfg['task']['scale'], shape=dataset.betas)

    contact_change_idx_all = find_contact_change_idx(dataset.contact)
    point_cloud, rgb= depth_to_pointcloud(cfg['task']['scene_rgbd'],fx=cfg['task']['focal_length'], fy=cfg['task']['focal_length'], scale=cfg['task']['depth_scale'], cx=int(cfg['task']['image_width']/2), cy=int(cfg['task']['image_height']/2)) # 720,1280,3
    
    trans_all = torch.zeros((dataset.contact.shape[0],3)) + transolver.foot_to_root_single(0,dataset.target_halpe_fil,dataset.pose,point_cloud)

    for i in range(0,len(contact_change_idx_all),2):
        if contact_change_idx_all[i][1] == True:
            pose_contact = dataset.pose[contact_change_idx_all[i][0]:contact_change_idx_all[i+1][0]+1,:]
            kp_2d_contact = dataset.target_halpe_fil[contact_change_idx_all[i][0]:contact_change_idx_all[i+1][0]+1,:,:]
            contact_contact = dataset.contact[contact_change_idx_all[i][0]:contact_change_idx_all[i+1][0]+1,:]
            pose, trans_contact = transolver.fix_trans_contact(pose_contact,kp_2d_contact,contact_contact)
            # ic(trans_contact)
            trans_all[contact_change_idx_all[i][0]:contact_change_idx_all[i+1][0]+1] = trans_contact + np.array([trans_all[contact_change_idx_all[i][0]-1][0],0,trans_all[contact_change_idx_all[i][0]-1][2]])

        elif contact_change_idx_all[i][1] == False:
            next_contact_frame  = contact_change_idx_all[i+1][0]+1
            if next_contact_frame >= dataset.contact.shape[0]:
                break
            last_contact_frame = contact_change_idx_all[i][0]-1
            root_src = transolver.foot_to_root_single(last_contact_frame, dataset.target_halpe_fil, dataset.pose, point_cloud)
            root_tar = transolver.foot_to_root_single(next_contact_frame, dataset.target_halpe_fil, dataset.pose, point_cloud)
            delta_depth = (root_tar - root_src)/(next_contact_frame-last_contact_frame)
            # ic(delta_depth)
            
            # Interpolation using parabolic approximation
            for idx in range(1,next_contact_frame-last_contact_frame):
                trans_all[idx+contact_change_idx_all[i][0]-1] = trans_all[contact_change_idx_all[i-1][0]] + idx*delta_depth
                trans_all[idx+contact_change_idx_all[i][0]-1,1] += (-9.8) /2 * (idx * 0.033)*(idx * 0.033 - (contact_change_idx_all[i+1][0]-contact_change_idx_all[i][0]+2)*0.033)
    
    return trans_all