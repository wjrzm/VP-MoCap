import torch
import numpy as np

from icecream import ic

from lib.initial_trans.transpose.config import *
from lib.initial_trans.transpose import transmath
from lib.initial_trans.transpose.model import ParametricModel

EPS = 1e-6

class TranSolver():
    def __init__(self, cfg,smpl_file='models\SMPL_NEUTRAL.pkl', floor=-1.20, scale=1.0, shape=None, hip_length=None, upper_leg_length=None, lower_leg_length=None, gravity_velocity=-1):

        # lower body joint
        m = ParametricModel(smpl_file, scale=scale)
        self.img_w = cfg['task']['image_width']
        self.img_h = cfg['task']['image_height']

        if shape is None:
            j, _ = m.get_zero_pose_joint_and_vertex(shape=shape)
        else:
            # shape (1, 10)
            # j (1, 24, 3)
            j, _ = m.get_zero_pose_joint_and_vertex(shape=torch.tensor(shape[0]))
            j = torch.squeeze(j)
        
        b = transmath.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = transmath.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        self.lower_body_bone = b
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0])
        # self.floor_y = j[7:12, 1].min().item()
        self.floor_y = floor

    @staticmethod
    def find_support_foot(target_halpe_fil, contact):
        kp_2d = target_halpe_fil
        # ic(kp_2d.shape)
        left_toe = (kp_2d[:,20,:] + kp_2d[:,22,:]) / 2
        right_toe = (kp_2d[:,21,:] + kp_2d[:,23,:]) / 2
        # use ankle instead of heel because heel is not privided in smpl
        # left_heel = kp_2d[:,24,:]
        # right_heel = kp_2d[:,25,:]
        left_ankle = kp_2d[:,15,:]
        right_ankle = kp_2d[:,16,:]

        vel_left_toe = left_toe[1:,:] - left_toe[:-1,:]
        vel_right_toe = right_toe[1:,:] - right_toe[:-1,:]
        # use ankle instead of heel because heel is not privided in smpl
        # vel_left_heel = left_heel[1:,:] - left_heel[:-1,:]
        # vel_right_heel = right_heel[1:,:] - right_heel[:-1,:]
        vel_left_ankle = left_ankle[1:,:] - left_ankle[:-1,:]
        vel_right_ankle = right_ankle[1:,:] - right_ankle[:-1,:]

        kp_save = np.zeros((left_toe.shape[0],2))
        index_save = [0]
        kp_full = np.stack((left_toe, left_ankle, right_toe, right_ankle),axis=1)
        for i in range(vel_left_toe.shape[0]):
            # left toe, left heel, right toe, right heel
            # if contact is 0, dis will be inf, so add EPS to avoid this
            dist_list = [np.linalg.norm(vel_left_toe[i,:])/(contact[i,0]+EPS),np.linalg.norm(vel_left_ankle[i,:])/(contact[i,1]+EPS),np.linalg.norm(vel_right_toe[i,:])/(contact[i,2]+EPS),np.linalg.norm(vel_right_ankle[i,:])/(contact[i,3]+EPS)]
            
            index_save.append(dist_list.index(min(dist_list)))
            kp_save[i+1,:] = kp_full[i+1,index_save[i+1],:]

        return index_save,kp_save
    
    def foot_to_root_single(self, frame, kp_2d, pose, point_cloud):
        """use left and right toe to calculate the root translation, we assume the two toes must be on the ground
        """
        left_toe = (kp_2d[frame,20,:] + kp_2d[frame,22,:]) / 2
        right_toe = (kp_2d[frame,21,:] + kp_2d[frame,23,:]) / 2

        pose = pose[frame]
        if type(pose) is np.ndarray:
            pose = torch.tensor(pose,dtype=torch.float32)
        if pose.ndim == 3:
            pose = pose.unsqueeze(0)

        j = transmath.forward_kinematics(pose[:, joint_set.lower_body],
                                        self.lower_body_bone.expand(pose.shape[0], -1, -1),
                                        joint_set.lower_body_parent)[1]
        
        left_toe_3d = point_cloud[int(left_toe[1]) if int(left_toe[1]) < self.img_h else self.img_h-1, int(left_toe[0]) if int(left_toe[0]) < self.img_w else self.img_w-1]
        right_toe_3d = point_cloud[int(right_toe[1]) if int(right_toe[1]) < self.img_h else self.img_h-1, int(right_toe[0]) if int(right_toe[0]) < self.img_w else self.img_w-1]

        foot_src_left = j[torch.arange(j.shape[0]), 7]
        foot_tar_left = torch.tensor(left_toe_3d,dtype=torch.float32)
        root_tar_left = foot_tar_left - foot_src_left

        foot_src_right = j[torch.arange(j.shape[0]), 8]
        foot_tar_right = torch.tensor(right_toe_3d,dtype=torch.float32)
        root_tar_right = foot_tar_right - foot_src_right

        # ic(root_tar_left,root_tar_right)
        root_tar = (root_tar_left + root_tar_right) / 2

        return root_tar

    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])
        
    def fix_trans_contact(self, pose, kp_2d, contact):

        if type(pose) is np.ndarray:
            pose = torch.tensor(pose,dtype=torch.float32)

        #? maybe kp_support_foot cannot present the root translation
        index, kp_support_foot = self.find_support_foot(kp_2d, contact)

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        j = transmath.forward_kinematics(pose[:, joint_set.lower_body],
                                        self.lower_body_bone.expand(pose.shape[0], -1, -1),
                                        joint_set.lower_body_parent)[1]
         
        vel_left_toe = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7]))
        vel_left_heel = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 5] - j[1:, 5]))
        vel_right_toe = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8]))
        vel_right_heel = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 6] - j[1:, 6]))

        vel_full = torch.stack((vel_left_toe, vel_left_heel, vel_right_toe, vel_right_heel), dim=1)
        vel_select = torch.stack([vel_full[i,index[i],:] for i in range(vel_full.shape[0])])
        velocity = self.gravity_velocity + vel_select
        # velocity = vel_select
        
        # remove penetration
        # ic(self.floor_y)
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + j[i, 5:9, 1].min().item()
            # ic(current_foot_y)
            if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                velocity[i, 1] = self.floor_y - current_foot_y
                # ic("Yes")
            current_root_y += velocity[i, 1].item()
            # ic(current_root_y)

        return pose, self.velocity_to_root_position(velocity)

