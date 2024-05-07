import os.path as osp
import imageio,cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh
from icecream import ic
import open3d as o3d

def depth2PointCloud(depth_map,fx,fy,cx,cy):
    z = depth_map
    r = np.arange(depth_map.shape[0])
    c = np.arange(depth_map.shape[1])
    x, y = np.meshgrid(c, r)
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z
    pointCloud = np.dstack([x, y, z])
    return pointCloud

#============================Floor============================
def normalizeFloor(floor_normal,floor_trans=([0,0,0]), target_normal=np.array([0, 1, 0])):    
    # to make the image look like the same as the original one
    # rotate 180 degree around the z axis
    angle = np.pi
    axis = np.array([0, 0, 1])
    aa = angle * axis
    rot = Rot.from_rotvec(aa)
    r2 = rot.as_matrix()
    floor_mat = np.eye(4)
    floor_mat[:3, :3] = r2
    floor_mat[:3, 3] = floor_trans
    
    return floor_mat

def calculateFloorNormal(point_cloud,xy, save_path=None):

    depth_slices = (point_cloud[xy[0]:xy[1], xy[2]:xy[3], :]).reshape([-1, 3])
    # trimesh.Trimesh(vertices=depth_slices,process=False).export('results/debug/depth_slice.obj')

    # open3d plane segmentation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth_slices)
    w, index = pcd.segment_plane(0.01,3,1000)
    depth_slices = np.asarray(pcd.select_by_index(index).points)
    
    # show depth slice by open3d
    # o3d.io.write_point_cloud('results/debug/depth_slice.ply', pcd.select_by_index(index))
    # o3d.visualization.draw_geometries([pcd.select_by_index(index)])

    A = depth_slices
    b = -1 * np.ones(depth_slices.shape[0])
    ATA, ATb = A.T @ A, A.T @ b
    ATA_inv = np.linalg.inv(ATA)
    x = np.dot(ATA_inv, ATb)
    floor_normal = x / np.linalg.norm(x)

    floor_rot = normalizeFloor(floor_normal)
    depth_slices_rot = (floor_rot[:3, :3] @ depth_slices.T + floor_rot[:3, 3].reshape([3, 1])).T
    floor_trans = np.array(
        [-np.median(depth_slices_rot[:, 0]), -np.median(depth_slices_rot[:, 1]), -np.median(depth_slices_rot[:, 2])])
    depth2floor = normalizeFloor(floor_normal=floor_normal, floor_trans=floor_trans)
    if save_path is not None:
        results = {
            'trans':floor_trans,
            'normal':floor_normal,
            'depth2floor':depth2floor
        }
        np.save(save_path, results)

    return floor_normal, floor_trans, depth2floor

def depth_to_pointcloud(depth_path,fx,fy, scale, cx, cy, save_path=None):
    """depth map to point cloud whose floor is on the XOZ plane

    Args:
        depth_path (str): the path of depth map
        xy (list, optional): _description_. Defaults to [520, 720, 0, 1280].
        fx (float, optional): _description_. Defaults to 608.074.
        fy (float, optional): _description_. Defaults to 608.244.
        cx (int, optional): _description_. Defaults to 640.
        cy (int, optional): _description_. Defaults to 360.
        save_path (str, optional): _description_. Defaults to None.

    Returns:
        np.array(720,1280,3): point cloud whose floor is on the XOZ plane
    """
    cx = int(cx)
    cy = int(cy)
    xy = [cy, cy*2, 0, cx*2]
    # load depth npy file 
    rgbd = np.load(depth_path,allow_pickle=True).item() # (720,1280)
    rgb = rgbd['rgb']
    depth = rgbd['depth'] * scale
    point_cloud = depth2PointCloud(depth,fx,fy,cx,cy)  # (720,1280,3)
    floor_normal, floor_trans, depth2floor = calculateFloorNormal(point_cloud,xy,save_path)
    point_cloud_fil = np.dot(depth2floor[:3, :3], point_cloud.reshape([-1, 3]).T).T

    return point_cloud_fil.reshape([cy*2,cx*2,3]) ,rgb
