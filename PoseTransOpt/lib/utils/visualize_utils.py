import torch
import cv2
import glob
import os
import os.path as osp
import numpy as np
import open3d as o3d

from icecream import ic
from tqdm import tqdm

def render_2d_keypoints(source,orig_img_bgr):
    cv2.circle(orig_img_bgr,(int(source[0]),int(source[1])),1,(0,0,255),1)
    return orig_img_bgr

class Camera():
    def __init__(self, cfg, dtype=torch.float32, device=None) -> None:
        self.img_w = cfg['task']['image_width']
        self.img_h = cfg['task']['image_height']
        self.focal_length = cfg['task']['focal_length']
        # self.focal_length = estimate_focal_length(self.img_h, self.img_w)

        self.camera_center = torch.tensor([self.img_w // 2, self.img_h // 2], dtype=dtype, device=device)
        self.rotation = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=dtype, device=device)
        self.translation = torch.tensor([0, 0, 0], dtype=dtype, device=device)

class Visualizer(Camera):
    def __init__(self, cfg, output_path, view='camera', fps=30, dtype=torch.float32, device=None):
        super().__init__(cfg, dtype, device)
        self.task_cfg = cfg['task']
        self.method_cfg = cfg['method']

        self.input_path_image = osp.join(self.task_cfg['input_path_base'], 'color')
        
        self.output_path = output_path
        self.smpl_file = 'models\SMPL_NEUTRAL.pkl'
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        
        self.fps = fps
        self.view = view

        self.img_all, self.img_path_list = self.load_image()
        self.scene_rgbd_path = self.task_cfg['scene_rgbd']
    
    def load_image(self):
        # get all image paths
        img_path_list = glob.glob(osp.join(self.input_path_image, '*.jpg'))
        img_path_list.extend(glob.glob(osp.join(self.input_path_image, '*.png')))
        img_path_list.extend(glob.glob(osp.join(self.input_path_image, '*.bmp')))
        img_path_list.sort()
        
        # load all images
        ic("Loading images ...")

        orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(img_path_list)]
        

        orig_img_bgr_all = orig_img_bgr_all[2:-2]
        img_path_list = img_path_list[2:-2]
        
        ic("Image number:", len(img_path_list))
        return orig_img_bgr_all, img_path_list
    
    def visual_smpl_2d_single(self, vertex, renderer, frame):
        # print(pred_vertices)
        pred_vert = vertex.detach().cpu().numpy()
        pred_vert = np.array(pred_vert)

        # img_all,img_path_list = self.load_image()
        orig_img_bgr = self.img_all[frame]

        # print(chosen_vert_opt_arr.shape)
        front_view_opt = renderer.render_front_view(pred_vert,bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
        # print(front_view_opt)
        filename_2dSMPL = "%03d_smpl_2d.png" % frame
        output_path_2dSMPl = osp.join(self.output_path, '2dSMPL')
        if not os.path.isdir(output_path_2dSMPl):
            os.mkdir(output_path_2dSMPl)
        front_view_path_2dSMPL = osp.join(output_path_2dSMPl, filename_2dSMPL)

        cv2.imwrite(front_view_path_2dSMPL, front_view_opt[:, :, ::-1])

        # delete the renderer for preparing a new one
        # renderer.delete()


    def create_o3d_mesh(self, verts, faces, colors=None):
        """
        Create a open3d mesh from vertices and faces.

        Parameters
        ----------
        verts : np.ndarray, shape [v, 3]
        Mesh vertices.
        faces : np.ndarray, shape [f, 3]
        Mesh faces.

        Returns
        -------
        o3d.geometry.TriangleMesh
        Open3d mesh.
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        # to remove the warning :[Open3D WARNING] Write OBJ can not include triangle normals.
        mesh.triangle_normals = o3d.utility.Vector3dVector([])
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()
        return mesh

    def save_o3d_mesh(self, verts, faces, frame):
        verts = verts.squeeze().detach().cpu().numpy()

        output_mesh = self.create_o3d_mesh(verts, faces)

        filename_3dSMPL = "%03d.obj" % frame
        output_path_3dSMPl = osp.join(self.output_path, '3dSMPL')
        if not os.path.isdir(output_path_3dSMPl):
            os.mkdir(output_path_3dSMPl)
        output_path = osp.join(output_path_3dSMPl, filename_3dSMPL)
        o3d.io.write_triangle_mesh(output_path, output_mesh, write_vertex_normals=False)
