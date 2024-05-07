import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def fil_pose(pose):
    pose_fil = []
    pose = np.squeeze(pose)

    for i in range(24):
        if pose.shape[1] == 72:
            q_origin = R.from_rotvec(pose[:,i*3:i*3+3]).as_quat()
        elif pose.shape[1] == 24:
            q_origin = R.from_matrix(pose[:,i]).as_quat()

        # fix the random jitter
        for i in range(1,q_origin.shape[0]):

            similarity = np.dot(q_origin[i],q_origin[i-1])
            if similarity < 0:
                q_origin[i,0] = -q_origin[i,0]
                q_origin[i,1] = -q_origin[i,1]
                q_origin[i,2] = -q_origin[i,2]
                q_origin[i,3] = -q_origin[i,3]

        WIDTH = 21
        POLY = 3
        q0 = savgol_filter(q_origin[:,0], WIDTH, POLY)
        q1 = savgol_filter(q_origin[:,1], WIDTH, POLY)
        q2 = savgol_filter(q_origin[:,2], WIDTH, POLY)
        q3 = savgol_filter(q_origin[:,3], WIDTH, POLY)

        temp = R.from_quat(np.stack((q0,q1,q2,q3),axis=1)).as_matrix()
        pose_fil.append(temp)

    pose_fil = np.stack(pose_fil,axis=1)
    
    return pose_fil

def fil_trans(trans):
    trans = np.squeeze(trans)
    trans_fil = np.zeros((trans.shape[0],3))

    trans_fil[:,0] = savgol_filter(trans[:,0], 15, 3)
    trans_fil[:,1] = savgol_filter(trans[:,1], 15, 3)
    trans_fil[:,2] = savgol_filter(trans[:,2], 15, 3)
    
    # trans_fil = torch.from_numpy(trans_fil).float()
    return trans_fil

def calc_border(target_halpe, target_halpe_select):
    BORDER_WIDTH = 3
    border_index_all = []
    for i in range(target_halpe.shape[1]):
        border_index = [0]
        for j in target_halpe_select[i]:
            if border_index == []:
                border_index.append(j[0]-BORDER_WIDTH if j[0]-BORDER_WIDTH >= 0 else 0)
                border_index.append(j[0]+BORDER_WIDTH if j[0]+BORDER_WIDTH <= target_halpe.shape[0] else target_halpe.shape[0])
            elif border_index[-1] < j[0]-BORDER_WIDTH:
                border_index.append(j[0]-BORDER_WIDTH if j[0]-BORDER_WIDTH >= 0 else 0)
                border_index.append(j[0]+BORDER_WIDTH if j[0]+BORDER_WIDTH <= target_halpe.shape[0] else target_halpe.shape[0])
            elif border_index[-1] >= j[0]-BORDER_WIDTH:
                border_index[-1] = j[0]+BORDER_WIDTH if j[0]+BORDER_WIDTH <= target_halpe.shape[0] else target_halpe.shape[0]
        if border_index[-1] < target_halpe.shape[0]:
            border_index.append(target_halpe.shape[0])
        border_index_all.append(border_index)
    
    # print(border_index_all)
    return border_index_all

def fil_2d_keypoints(target_halpe, target_halpe_score, target_halpe_fil):
    target_halpe_select = []
            
    for i in range(target_halpe.shape[1]):
        temp = []
        for j in range(target_halpe.shape[0]):
            if target_halpe_score[j][i] < 0.5:
                temp.append([j, target_halpe[j][i]])
        
        
        target_halpe_select.append(temp)
    
    border_index = calc_border(target_halpe, target_halpe_select)

    # use savgol_filter
    
    EXTEND = 10
    for i in range(target_halpe.shape[1]):
        FLAG = 1
        for j in range(len(border_index[i]) - 1):
            
            if FLAG == 1:
                target_halpe_fil[border_index[i][j]:border_index[i][j+1],i,0] = target_halpe[border_index[i][j]:border_index[i][j+1],i,0]
                target_halpe_fil[border_index[i][j]:border_index[i][j+1],i,1] = target_halpe[border_index[i][j]:border_index[i][j+1],i,1]
                FLAG = 0
            elif FLAG == 0:
                if border_index[i][j+1]+EXTEND < target_halpe.shape[0]:
                    target_halpe_fil[border_index[i][j]:border_index[i][j+1],i,0] = savgol_filter(target_halpe[border_index[i][j]-EXTEND:border_index[i][j+1]+EXTEND,i,0], 21, 2)[EXTEND:-EXTEND]
                    target_halpe_fil[border_index[i][j]:border_index[i][j+1],i,1] = savgol_filter(target_halpe[border_index[i][j]-EXTEND:border_index[i][j+1]+EXTEND,i,1], 21, 2)[EXTEND:-EXTEND]
                    FLAG = 1
                elif border_index[i][j+1]+EXTEND >= target_halpe.shape[0]:
                    target_halpe_fil[border_index[i][j]:,i,0] = savgol_filter(target_halpe[border_index[i][j]-EXTEND:,i,0], 11, 2)[EXTEND:]
                    target_halpe_fil[border_index[i][j]:,i,1] = savgol_filter(target_halpe[border_index[i][j]-EXTEND:,i,1], 11, 2)[EXTEND:]
                    FLAG = 1
    
    return target_halpe_fil
