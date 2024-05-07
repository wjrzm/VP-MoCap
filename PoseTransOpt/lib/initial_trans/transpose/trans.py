import torch.nn
from torch.nn.functional import relu
from lib.transpose.config import *
from lib.transpose import transmath
from lib.transpose.model import ParametricModel
import numpy as np
from icecream import ic

class TransSolver(torch.nn.Module):
    r"""
    Whole pipeline for translation estimation.
    """
    def __init__(self, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.0018):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()

        # lower body joint
        m = ParametricModel(paths.smpl_file)
        j, _ = m.get_zero_pose_joint_and_vertex()
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

        # constant
        self.global_to_local_pose = m.inverse_kinematics_R
        self.lower_body_bone = b
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0])
        self.floor_y = j[7:12, 1].min().item()
        # ic(j[:,1])


    @torch.no_grad()
    def forward_offline(self, pose, index):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and translation tensor in shape [num_frame, 3].
        """

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        j = transmath.forward_kinematics(pose[:, joint_set.lower_body],
                                        self.lower_body_bone.expand(pose.shape[0], -1, -1),
                                        joint_set.lower_body_parent)[1]
         
        # ic(j[0])

        vel_left_toe = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7]))
        vel_left_heel = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 5] - j[1:, 5]))
        vel_right_toe = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8]))
        vel_right_heel = torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 6] - j[1:, 6]))
        vel_full = torch.stack((vel_left_toe, vel_left_heel, vel_right_toe, vel_right_heel), dim=1)
        vel_select = torch.stack([vel_full[i,index[i],:] for i in range(vel_full.shape[0])])
        # ic(vel_select.shape)
        velocity = self.gravity_velocity + vel_select
        # velocity = vel_select
        

        # remove penetration
        ic(self.floor_y)
        current_root_y = 0
        for i in range(velocity.shape[0]):
            # ic(i)
            current_foot_y = current_root_y + j[i, 5:9, 1].min().item()
            # ic(current_foot_y)
            if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                velocity[i, 1] = self.floor_y - current_foot_y
                # ic("Yes")
            current_root_y += velocity[i, 1].item()
            # ic(current_root_y)
            

        return pose, self.velocity_to_root_position(velocity)


    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])
