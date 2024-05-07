import glob
import os.path as osp
import logging
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from lib.utils.depth_utils import depth_to_pointcloud

from icecream import ic
from tqdm import tqdm

log = logging.getLogger(__name__)
class Dataset():
    def __init__(self, cfg):
        self.task_cfg = cfg['task']
        self.method_cfg = cfg['method']
        
        self.input_path_base = self.task_cfg['input_path_base']
        self.input_path_image = osp.join(self.input_path_base, 'color')
        self.input_path_RTMPose = osp.join(self.input_path_base, 'keypoints')
        self.input_path_npz = glob.glob(osp.join(self.input_path_base, '*.npz'))[0]
        
        self.input_path_template_scene_rgbd = self.task_cfg['scene_rgbd']
        self.point_cloud, _= depth_to_pointcloud(self.input_path_template_scene_rgbd, fx=cfg['task']['focal_length'], fy=cfg['task']['focal_length'], scale=cfg['task']['depth_scale'], cx=int(cfg['task']['image_width']/2), cy=int(cfg['task']['image_height']/2))

        self.pose_filter = self.method_cfg['pose_filter']
        self.kp_filter = self.method_cfg['kp_filter']
        self.kp_score = self.method_cfg['kp_score']

        self.input_path_pred_contact = osp.join(self.input_path_base, 'pred_contact_smpl')
        self.target_halpe_fil, self.target_halpe_score = self.load_2d_keypoints()
        self.pose,self.betas,self.trans_origin = self.load_pose()
        self.thres_contact = self.task_cfg['thres_contact']
        self.contact, self.vertex_contact = self.load_pred_contact_smpl()
        
    def load_pred_contact_smpl(self):
            pred_contact_path_list = glob.glob(osp.join(self.input_path_pred_contact, '*.npy'))
            pred_contact_path_list.sort()

            ic("Loading pred_contact_smpl npy ...")

            contact_flag_all = []
            vertex_contact_all = []
            for pred_contact_path in tqdm(pred_contact_path_list):
                vertex_contact_prob = np.load(pred_contact_path,allow_pickle=True).item()['contact_smpl']['pred']
                vertex_contact = np.zeros_like(vertex_contact_prob)

                for i in range(vertex_contact_prob.shape[1]):
                    if vertex_contact_prob[0,i] > 0.5:
                        vertex_contact[0,i] = 1
                    if vertex_contact_prob[1,i] > 0.5:
                        vertex_contact[1,i] = 1
                
                    
                LT_contact = np.sum(vertex_contact[0,:48])
                RT_contact = np.sum(vertex_contact[1,:48])
                LH_contact = np.sum(vertex_contact[0,48:])
                RH_contact = np.sum(vertex_contact[1,48:])

                _contact_flag = np.zeros(4)

                # print(LT_contact, LH_contact, RT_contact, RH_contact)
                if LT_contact > self.thres_contact:
                    _contact_flag[0] = 1
                else:
                    vertex_contact[0,:48] = np.zeros_like(vertex_contact[0,:48])
                if LH_contact > self.thres_contact:
                    _contact_flag[1] = 1
                else:
                    vertex_contact[0,48:] = np.zeros_like(vertex_contact[0,48:])
                if RT_contact > self.thres_contact:
                    _contact_flag[2] = 1
                else:
                    vertex_contact[1,:48] = np.zeros_like(vertex_contact[1,:48])
                if RH_contact > self.thres_contact:
                    _contact_flag[3] = 1
                else:
                    vertex_contact[1,48:] = np.zeros_like(vertex_contact[1,48:])

                contact_flag_all.append(_contact_flag)
                vertex_contact_all.append(vertex_contact)

                # print(_contact_flag)
                
                # show image
                # img_left = drawSMPLFoot_contact(self.v_footR,self.footIdsR,np.array(self.model_temp.faces),
                #                 vert_contact=vertex_contact[0],save_name=None,
                #                 point_size=4)
                
                # img_right = drawSMPLFoot_contact(self.v_footL,self.footIdsL,np.array(self.model_temp.faces),
                #                 vert_contact=vertex_contact[1],save_name=None,
                #                 point_size=4)
                
                # img = np.concatenate([img_left,img_right],axis=1)
                # img = cv2.putText(img, f"{_contact_flag}", (int(50), int(50)),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                # output_path = pred_contact_path[:-4] + '_vertex.png'
                # cv2.imwrite(output_path ,img)

            return np.array(contact_flag_all), np.array(vertex_contact_all)

    def load_pose(self):
        npz = np.load(self.input_path_npz)
        
        pred_betas = npz['shape']
        pred_rotvec = npz['pose']
        pred_cam_full = npz['global_t']

        # all betas are the same
        # pred_betas = pred_betas[0].reshape(1,10).repeat(pred_betas.shape[0], axis=0)
        pred_betas = np.mean(pred_betas, axis=0).reshape(1,10).repeat(pred_betas.shape[0], axis=0)

        if self.pose_filter:
            if pred_rotvec.shape[1] != 72:
                raise ValueError('The shape of pred_rotvec is not 72')
            pred_rotmat_fil = []
            for i in range(0,72,3):
                q_origin = R.from_rotvec(pred_rotvec[:,i:i+3]).as_quat()
                if i == 0:
                    for i in range(1,q_origin.shape[0]):
                        similarity = np.dot(q_origin[i],q_origin[i-1])
                        if similarity < 0:
                            q_origin[i,0] = -q_origin[i,0]
                            q_origin[i,1] = -q_origin[i,1]
                            q_origin[i,2] = -q_origin[i,2]
                            q_origin[i,3] = -q_origin[i,3]
                        similarity = np.dot(q_origin[i],q_origin[i-1])

                        if similarity < 0.5:
                            # print('FIND SIMILARITY LOW!!!!!')
                            # print('USING BEFORE DATA!!!!!')
                            # file = open("similarity.txt", "a+")
                            # file.write(f"{self.input_path_base} {i} {similarity}\n")
                            # file.close()
                            q_origin[i,0] = q_origin[i-1,0]
                            q_origin[i,1] = q_origin[i-1,1]
                            q_origin[i,2] = q_origin[i-1,2]
                            q_origin[i,3] = q_origin[i-1,3]
                q0 = savgol_filter(q_origin[:,0], 17, 5)
                q1 = savgol_filter(q_origin[:,1], 17, 5)
                q2 = savgol_filter(q_origin[:,2], 17, 5)
                q3 = savgol_filter(q_origin[:,3], 17, 5)
                temp = R.from_quat(np.stack((q0,q1,q2,q3),axis=1)).as_matrix()
                pred_rotmat_fil.append(temp)
            pred_rotmat_fil = np.stack(pred_rotmat_fil,axis=1)
            pred_rotmat = pred_rotmat_fil

            trans0 = savgol_filter(pred_cam_full[:,0], 17, 5)
            trans1 = savgol_filter(pred_cam_full[:,1], 17, 5)
            trans2 = savgol_filter(pred_cam_full[:,2], 17, 3)

            pred_cam_full_fill = np.stack((trans0,trans1,trans2),axis=1)
            pred_cam_full = pred_cam_full_fill
        elif self.pose_filter == False:
            # ic('no filter')
            pred_rotmat = []
            for i in range(0,72,3):
                pred_rotmat.append(R.from_rotvec(pred_rotvec[:,i:i+3]).as_matrix())
            pred_rotmat = np.stack(pred_rotmat,axis=1)
            
        # flip x-axis
        # ic(pred_rotmat.shape)
        pred_rotmat[:,0] = R.from_rotvec([np.pi,0,0]).as_matrix() @ pred_rotmat[:,0]
        pred_rotmat[:,0] = R.from_rotvec([0,np.pi,0]).as_matrix() @ pred_rotmat[:,0]
        
        pred_rotmat = pred_rotmat[2:-2]
        pred_betas = pred_betas[2:-2]
        pred_cam_full = pred_cam_full[2:-2]

        return pred_rotmat,pred_betas,pred_cam_full

    def load_2d_keypoints(self):
        kp_2d_path = self.input_path_RTMPose
        kp_2d_path_list = glob.glob(osp.join(kp_2d_path, '*.npy'))
        kp_2d_path_list.sort()
        ic("Loading 2d keypoints ...")
        kp_2d_all = [np.load(kp_2d_path, allow_pickle=True).item() for kp_2d_path in tqdm(kp_2d_path_list)]

        target_halpe = np.array([kp_2d_all[i]['keypoints'] for i in range(len(kp_2d_all))])
        target_halpe_score = np.array([kp_2d_all[i]['keypoint_scores'] for i in range(len(kp_2d_all))])
                
        if self.kp_filter:
            target_halpe_fil = np.zeros_like(target_halpe)

            # use simple savgol_filter 
            for i in range(target_halpe.shape[1]):
                target_halpe_fil[:,i,0] = savgol_filter(target_halpe[:,i,0], 15, 3)
                target_halpe_fil[:,i,1] = savgol_filter(target_halpe[:,i,1], 15, 3)
            
        elif self.kp_filter == False:
            target_halpe_fil = target_halpe

        # No score
        if self.kp_score == False:
            target_halpe_score = np.ones_like(target_halpe_score)
        # use savgol_filter

        target_halpe_fil = target_halpe_fil[2:-2]
        target_halpe_score = target_halpe_score[2:-2]

        return target_halpe_fil, target_halpe_score