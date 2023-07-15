'''
Author: wugaoqiang wugaoqiang@lixiang.com
Date: 2023-07-14 20:02:30
LastEditors: wugaoqiang wugaoqiang@lixiang.com
LastEditTime: 2023-07-14 20:05:16
FilePath: /SurroundDepth/tools/dispaly_point_cloud.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
# import mmcv
from collections import OrderedDict
from glob import glob
import cv2
import argparse


def save_view_point_clound(pc_path_list, json_path):
    # Load xyzrgb numpy data
    point_post = np.array([])
    point_colors = np.array([])
    for camera_id, pc_path in pc_path_list[1].items():
        #for key,value in pc_path
        pc_data = np.load(pc_path)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=2000, height=1500)

        # Transpose the matrix to match the Open3D convention
        data = np.transpose(pc_data)
        # Set point cloud properties
        print(data[:, :3][0])
        if point_post.size == 0 or point_colors.size == 0:
            point_post = data[:, :3]
            point_colors = data[:, 3:]
        else:
            point_post = np.vstack((point_post, data[:, :3]))
            point_colors = np.vstack((point_colors, data[:, 3:]))
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_post)  # Extract xyz coordinates
    pcd.colors = o3d.utility.Vector3dVector(
        point_colors)  # Extract RGB color values and normalize to range [0,1]

    # Visualize the point cloud
    #o3d.visualization.draw_geometries([pcd])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry(pcd)
    vis.add_geometry(axis_pcd)
    vis.run()  # user changes the view and press "q" to terminate

    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(json_path, param)

    vis.destroy_window()

def vis_autolabel_multi(label_path, json_path, gt_path=None, cut_range=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=2000, height=1500)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True

    param = o3d.io.read_pinhole_camera_parameters(json_path)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    to_reset = True

    origin = (0, -24, -1)
    voxel_size = 0.3

    pred_files = glob(label_path + "/*.labels")
    gt_files = glob(gt_path + "/*.npz")
    # gt_files = [os.path.join(gt_occ_path, os.path.basename(p)) for p in gt_occ_list]

    pred_dic = dict()
    gt_dic = dict()
    for path in pred_files:
        filename = os.path.basename(path)
        # TODO(wgq)
        start_index = filename.find('_') + 1
        timestamp = filename[start_index:start_index + 13]
        pred_dic[timestamp] = path

    for path in gt_files:
        filename = os.path.basename(path)
        start_index = filename.rfind('_') + 1
        end_index = filename.rfind('.')
        timestamp = filename[start_index:end_index]
        gt_dic[timestamp] = path

    gt_list = list()
    pred_list = list()
    pred_dic_sort = sorted(pred_dic.items(), key=lambda x: int(x[0]))
    for pred_pair in pred_dic_sort:
        for g_key, g_value in gt_dic.items():
            p_key_i = int(pred_pair[0])
            g_key_i = int(g_key)
            if (p_key_i > g_key_i - 10) and (p_key_i < g_key_i + 10):
                gt_list.append(g_value)
                pred_list.append(pred_pair[1])

    for p, g in zip(pred_list, gt_list):
        vg = get_geo_autolabel(p, (12, 160, 264), origin, voxel_size, cut_range=cut_range) #
        vis.clear_geometries()

        vis.add_geometry(vg)
        axis = o3d.geometry.TriangleMesh().create_coordinate_frame(size=2, origin=np.array([0, 0., 0]))
        vis.add_geometry(axis)

        vg_gt = get_gt_autolabel_mask(g, (334, 160, 16), origin, voxel_size, cut_range) #
        vis.add_geometry(vg_gt)

        # cam3_img, pre = add_single_img(gt_path, clip_name, token,mark)
        # if len(pre)!= 0:
        #     mark.append(pre)
        # vis.add_geometry(cam3_img)
        # ctr.convert_from_pinhole_camera_parameters(param)
        time.sleep(0.1)

        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()


def data_files_preprocess(point_files: str):
    if len(point_files) == 0:
        exit()
    data_frame_paths = {}
    for pc_path in point_files:
        file_dic = dict
        parts = pc_path.split("_")
        frame_id = int(parts[3])
        camera_id = int(parts[4].split(".")[0])
        file_dic = {camera_id: pc_path}
        if frame_id not in data_frame_paths.keys():
            data_frame_paths[frame_id] = file_dic
        else:
            data_frame_paths[frame_id][camera_id] = pc_path
    pred_dic_sort = sorted(data_frame_paths.items(), key=lambda x: int(x[0]))
    return pred_dic_sort



parser = argparse.ArgumentParser()
parser.add_argument('--pc_folder', type=str, default='/home/wugaoqiang/work/depth/SurroundDepth/data/nuscenes/pred_points')
parser.add_argument('--img_folder', type=str, default='/home/wugaoqiang/work/depth/SurroundDepth/data/nuscenes/pred_img')
parser.add_argument('--enable_img_display', type=bool, default=False)

if __name__ =="__main__":
    args = parser.parse_args()
    pc_folder = args.pc_folder
    img_folder = args.img_folder
    enable_img_display = args.enable_img_display
    point_files = glob(pc_folder + "/*.npy")
    if enable_img_display:
        pred_files = glob(img_folder + "/*.jpg")
    data_frame_files = data_files_preprocess(point_files)

    if len(data_frame_files) > 0:
        set_debug_view_data_path = data_frame_files[-1]
        json_path = './autolabel_view3.json'
        # save_view_point_pred_occ(set_debug_view_data_path, json_path, None)
        save_view_point_clound(set_debug_view_data_path, json_path)
        # vis_autolabel_multi(pred_path, json_path, gt_path, cut_range)