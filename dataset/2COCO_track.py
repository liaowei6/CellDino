import json
import glob
import cv2 
import os
import torch
import numpy as np
import tifffile
import re

def convert2xywh(box):
    
    x = box[1]
    y = box[2]
    w = box[3] - box[1]
    h = box[4] - box[2]
    box_new = [float(x), float(y), float(w), float(h)]
    return box_new


def convert(box):
    x = box[1]
    y = box[2]
    w = box[3]
    h = box[4]
    t = box[5]
    box_new = [float(x), float(y), float(w), float(h), float(t)]
    return box_new



def get_rotated_rect_vertices(rect):
    """
    计算旋转矩形的 4 个顶点坐标
    Args:
        rect: 1D array [x, y, w, h, theta] (theta in degrees)
    Returns:
        vertices: 4x2 array, 4 vertices of the rotated rectangle
    """
    x, y, w, h, theta = rect
    theta_rad = np.deg2rad(theta)  # 角度转弧度
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # 计算半宽和半高
    half_w = w / 2
    half_h = h / 2

    # 定义 4 个顶点（相对于中心点）
    vertices_rel = np.array([
        [-half_w, -half_h],  # 左下
        [half_w, -half_h],   # 右下
        [half_w, half_h],    # 右上
        [-half_w, half_h]    # 左上
    ])  # shape: (4, 2)

    # 旋转矩阵 (2x2)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # 旋转顶点 (4,2) = (4,2) @ (2,2)
    rotated_vertices = vertices_rel @ rotation_matrix.T

    # 平移回原坐标系
    rotated_vertices += np.array([x, y])

    return rotated_vertices

# def rbb2hbb(rect):
#     """
#     获取旋转矩形的最小外接矩形 (AABB)
#     Args:
#         rect: 1D array [x, y, w, h, theta]
#     Returns:
#         aabb: [x_min, y_min, width, height]
#     """
#     vertices = get_rotated_rect_vertices(rect)  # (4, 2)
#     x_min, y_min = np.min(vertices, axis=0)
#     x_max, y_max = np.max(vertices, axis=0)

#     return [x_min, y_min, x_max - x_min, y_max - y_min]

import math
def rbb2hbb(rect):
    x, y, w, h, theta = rect
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # 计算旋转后的四个顶点坐标
    dx = np.array([-w/2, w/2, w/2, -w/2])
    dy = np.array([-h/2, -h/2, h/2, h/2])
    rotated_x = x + dx * cos_theta - dy * sin_theta
    rotated_y = y + dx * sin_theta + dy * cos_theta

    # 计算外接矩形
    min_x, max_x = np.min(rotated_x), np.max(rotated_x)
    min_y, max_y = np.min(rotated_y), np.max(rotated_y)

    return [min_x, min_y, max_x - min_x, max_y - min_y]

class tococo(object):
    def __init__(self, data_path, save_path):
        self.images = []
        self.categories = []
        self.annotations = []
        # 返回每张图片的地址
        self.data_path = data_path
        self.save_path = save_path
        # 可根据情况设置类别，这里只设置了一类
        self.class_ids = {'cell': 1, "background": 2}
        self.class_id = 1
        self.coco = {}
 
    def npz_to_coco_val(self):
        annid = 0
        sub_dirs = ["01", "02"]
        num = 0
        for path in self.data_path:
            #for dir in sub_dirs:
            for i in range(91):
            #for dir in sub_dirs:
                dir = f'batch_{i}'
                images = [
                    (image_file.split(".")[0], image_file, )
                    for image_file in os.listdir(path + dir + "/images")
                    ]
                masks = [
                    (mask_file.split(".")[0], mask_file)
                    for mask_file in os.listdir(path + dir + "/masks")
                    ]
                boxes_order = [
                    (int(box_file.split(".")[0].split("_")[0]), box_file)
                    for box_file in os.listdir(path + dir + "/bounding_boxes")
                    ]
                instance_masks_order = [
                    (instance_mask_file.split(".")[0], instance_mask_file)
                    for instance_mask_file in os.listdir(path + dir + "/instance_mask")
                    ]
                boxes_order.sort(key=lambda x: x[0])
                instance_masks_order.sort(key=lambda x: x[0])
                boxes = []
                instance_mask = []
                pre_indice = None
                box_item = []
                mask_item = []
                for box_file, mask_file in zip(boxes_order, instance_masks_order):
                    box_indice = box_file[0]
                    box_id = int(box_file[1].split(".")[0].split("_")[1])
                    mask_id = (int(mask_file[1].split(".")[0].split("_")[1]))
                    if pre_indice == None or box_indice == pre_indice:
                        pre_indice = box_indice
                        box_item.append((box_id, box_file[1]))
                        mask_item.append((mask_id, mask_file[1]))
                        continue
                    boxes.append((pre_indice, box_item))
                    instance_mask.append((pre_indice, mask_item))
                    box_item = []
                    mask_item = []
                    pre_indice = box_indice
                    box_item.append((box_id, box_file[1]))
                    mask_item.append((mask_id, mask_file[1]))
                boxes.append((pre_indice, box_item))
                instance_mask.append((pre_indice, mask_item))

                images.sort(key=lambda x: x[0])
                masks.sort(key=lambda x: x[0])
                boxes.sort(key=lambda x: x[0])
                instance_mask.sort(key=lambda x: x[0])
                length = len(images)
                for i in range(length):
                    image_path = path + dir + "/images/" + images[i][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    mask = masks[i][1]
                    box_set = boxes[i][1]
                    instance_mask_set = instance_mask[i][1]
                    box_set.sort(key=lambda x:x[0])
                    instance_mask_set.sort(key=lambda x:x[0])
                    instance_num = len(boxes[i][1])
                    for j in range(instance_num):
                        self.annotations.append(self.get_annotations(box_set[j], i + num, annid, 1, mask, instance_mask_set[j], path + dir))
                        annid += 1           
                    self.images.append(self.get_images(image_path, h, w, i + num))
                num = num + i + 1
        self.coco["images"] = self.images
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations
        # print(self.coco)
    
    def npz_to_coco_train(self):
        annid = 0
        sub_dirs = ["01", "02"]
        num = 0
        for path in self.data_path:
            for dir in sub_dirs:
                images = [
                    (image_file.split(".")[0], image_file, )
                    for image_file in os.listdir(path + dir + "/images")
                    ]
                masks = [
                    (mask_file.split(".")[0], mask_file)
                    for mask_file in os.listdir(path + dir + "/masks")
                    ]
                boxes_order = [
                    (int(box_file.split(".")[0].split("_")[0]), box_file)
                    for box_file in os.listdir(path + dir + "/bounding_boxes")
                    ]
                instance_masks_order = [
                    (instance_mask_file.split(".")[0], instance_mask_file)
                    for instance_mask_file in os.listdir(path + dir + "/instance_mask")
                    ]
                with open(path + dir + '/lineage.txt', 'r') as file:
                    # 按行读取文本文件
                    lineage = file.readlines()
                
                boxes_order.sort(key=lambda x: x[0])
                instance_masks_order.sort(key=lambda x: x[0])
                boxes = []
                instance_mask = []
                pre_indice = None
                box_item = []
                mask_item = []
                for box_file, mask_file in zip(boxes_order, instance_masks_order):
                    box_indice = box_file[0]
                    box_id = int(box_file[1].split(".")[0].split("_")[1])
                    mask_id = (int(mask_file[1].split(".")[0].split("_")[1]))
                    if pre_indice == None or box_indice == pre_indice:
                        pre_indice = box_indice
                        box_item.append((box_id, box_file[1]))
                        mask_item.append((mask_id, mask_file[1]))
                        continue
                    boxes.append((pre_indice, box_item))
                    instance_mask.append((pre_indice, mask_item))
                    box_item = []
                    mask_item = []
                    pre_indice = box_indice
                    box_item.append((box_id, box_file[1]))
                    mask_item.append((mask_id, mask_file[1]))
                boxes.append((pre_indice, box_item))
                instance_mask.append((pre_indice, mask_item))

                images.sort(key=lambda x: x[0])
                masks.sort(key=lambda x: x[0])
                boxes.sort(key=lambda x: x[0])
                instance_mask.sort(key=lambda x: x[0])
                length = len(images)
                non = 0
                total_i =0
                for i in range(length):
                    #首张图片
                    track_intervals = min(i + 3, length -1)
                    image_path = path + dir + "/images/" + images[i][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    mask = masks[i][1]
                    box_set = boxes[i][1]
                    instance_mask_set = instance_mask[i][1]
                    box_set.sort(key=lambda x:x[0])
                    instance_mask_set.sort(key=lambda x:x[0])
                    instance_num = len(instance_mask_set)  
                    for ii in range(i +1, track_intervals + 1):     
                        for j in range(instance_num):
                            self.annotations.append(self.get_annotations_track(box_set[j], total_i + num, annid, 1, instance_mask_set[j], path + dir, first=True))
                            annid += 1    
                        #后续图片
                        image_path_second = path + dir + "/images/" + images[ii][1]
                        box_set_second = boxes[ii][1]
                        instance_mask_set_second = instance_mask[ii][1]
                        box_set_second.sort(key=lambda x:x[0])
                        instance_mask_set_second.sort(key=lambda x:x[0])
                        instance_num_second = len(instance_mask_set_second) 
                        for jj in range(instance_num_second):
                            self.annotations.append(self.get_annotations_track(box_set_second[jj], total_i + num, annid, 1, instance_mask_set_second[jj], path + dir, first=False))
                            annid += 1 
                        #处理分裂时间 Fluo-N2DH-GOWT1数据集看不出分裂固在训练时省略该步骤
                        moists = {}
                        # for m in lineage:
                        #     m = m.strip().split(" ")
                        #     if int(m[3]) > 0:
                        #         if i < int(m[1]) <= ii:
                        #             if int(m[3]) not in moists:
                        #                 moists[int(m[3])] = []
                        #             moists[int(m[3])].append(int(m[0]))
                        self.images.append(self.get_images_track(image_path, image_path_second, h, w, total_i + num, moists))
                        total_i += 1 

                num = num + total_i + 1 - non
        self.coco["images"] = self.images
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations

    def npz_to_coco_train_simi(self):
        annid = 0
        sub_dirs = ["01", "02"]
        num = 0
        for path in self.data_path:
            for i in range(91):
            #for dir in sub_dirs:
                dir = f'batch_{i}'
                images = [
                    (image_file.split(".")[0], image_file, )
                    for image_file in os.listdir(path + dir + "/images")
                    ]
                masks = [
                    (mask_file.split(".")[0], mask_file)
                    for mask_file in os.listdir(path + dir + "/masks")
                    ]
                boxes_order = [
                    (int(box_file.split(".")[0].split("_")[0]), box_file)
                    for box_file in os.listdir(path + dir + "/bounding_boxes")
                    ]
                instance_masks_order = [
                    (instance_mask_file.split(".")[0], instance_mask_file)
                    for instance_mask_file in os.listdir(path + dir + "/instance_mask")
                    ]
                with open(path + dir + '/lineages.txt', 'r') as file:
                    # 按行读取文本文件
                    lineage = file.readlines()
                
                boxes_order.sort(key=lambda x: x[0])
                instance_masks_order.sort(key=lambda x: x[0])
                boxes = []
                instance_mask = []
                pre_indice = None
                box_item = []
                mask_item = []
                for box_file, mask_file in zip(boxes_order, instance_masks_order):
                    box_indice = box_file[0]
                    box_id = int(box_file[1].split(".")[0].split("_")[1])
                    mask_id = (int(mask_file[1].split(".")[0].split("_")[1]))
                    if pre_indice == None or box_indice == pre_indice:
                        pre_indice = box_indice
                        box_item.append((box_id, box_file[1]))
                        mask_item.append((mask_id, mask_file[1]))
                        continue
                    boxes.append((pre_indice, box_item))
                    instance_mask.append((pre_indice, mask_item))
                    box_item = []
                    mask_item = []
                    pre_indice = box_indice
                    box_item.append((box_id, box_file[1]))
                    mask_item.append((mask_id, mask_file[1]))
                boxes.append((pre_indice, box_item))
                instance_mask.append((pre_indice, mask_item))

                images.sort(key=lambda x: x[0])
                masks.sort(key=lambda x: x[0])
                boxes.sort(key=lambda x: x[0])
                instance_mask.sort(key=lambda x: x[0])
                length = len(images)
                non = 0
                total_i =0
                sample_intervals = 5   #采样间隔，即每隔几帧采样一个全标注样本
                intervals = 1
                for i in range(0, length, intervals):
                    #首张图片
                    track_end = min(i + sample_intervals, length -1)
                    track_begin = max(0, i-sample_intervals)
                    image_path = path + dir + "/images/" + images[i][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    mask = masks[i][1]
                    box_set = boxes[i][1]
                    instance_mask_set = instance_mask[i][1]
                    box_set.sort(key=lambda x:x[0])
                    instance_mask_set.sort(key=lambda x:x[0])
                    instance_num = len(instance_mask_set)  
                    for ii in range(track_begin, track_end + 1):     
                        for j in range(instance_num):
                            self.annotations.append(self.get_annotations_track(box_set[j], total_i + num, annid, 1, instance_mask_set[j], path + dir, first=True))
                            annid += 1    
                        #后续图片
                        image_path_second = path + dir + "/images/" + images[ii][1]
                        box_set_second = boxes[ii][1]
                        instance_mask_set_second = instance_mask[ii][1]
                        box_set_second.sort(key=lambda x:x[0])
                        instance_mask_set_second.sort(key=lambda x:x[0])
                        instance_num_second = len(instance_mask_set_second) 
                        for jj in range(instance_num_second):
                            self.annotations.append(self.get_annotations_track(box_set_second[jj], total_i + num, annid, 1, instance_mask_set_second[jj], path + dir, first=False))
                            annid += 1 
                        #处理分裂时间 Fluo-N2DH-GOWT1数据集看不出分裂固在训练时省略该步骤
                        moists = {}
                        # for m in lineage:
                        #     m = m.strip().split(" ")
                        #     if int(m[3]) > 0:
                        #         if i < int(m[1]) <= ii:
                        #             if int(m[3]) not in moists:
                        #                 moists[int(m[3])] = []
                        #             moists[int(m[3])].append(int(m[0]))
                        self.images.append(self.get_images_track(image_path, image_path_second, h, w, total_i + num, moists))
                        total_i += 1 

                num = num + total_i + 1 - non
        self.coco["images"] = self.images
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations

    def npz_to_coco_simi(self):
        annid = 0
        sub_dirs = ["01", "02"]
        num = 0
        for path in self.data_path:
            for dir in sub_dirs:
                images = [
                    (image_file.split(".")[0], image_file, )
                    for image_file in os.listdir(path + dir + "/images")
                    ]
                boxes_order = [
                    (int(box_file.split(".")[0].split("_")[0]), box_file)
                    for box_file in os.listdir(path + dir + "/bounding_boxes")
                    ]
                instance_masks_order = [
                    (instance_mask_file.split(".")[0], instance_mask_file)
                    for instance_mask_file in os.listdir(path + dir + "/instance_mask")
                    ]
                coords_order = []
                for instance_mask_file in os.listdir(path + dir + "/coords"):
                    # 使用正则表达式匹配文件名
                    match = re.match(r'man_track(\d+)_value_(\d+)\.txt', instance_mask_file)
                    if match:
                        # 提取匹配的数字部分
                        frame_number = match.group(1)
                        value_number = match.group(2)
                        # 构建新的文件名
                        new_filename = f'{frame_number}_{value_number}'
                        # 将新的文件名和原始文件名组成元组
                        coords_order.append((new_filename, instance_mask_file))
                boxes_order.sort(key=lambda x: x[0])
                instance_masks_order.sort(key=lambda x: x[0])
                coords_order.sort(key=lambda x: x[0])
                coords = []
                boxes = []
                instance_mask = []
                pre_indice = None
                coord_item = []
                box_item = []
                mask_item = []
                indice_list = []
                #按帧处理GT中的边框和掩码
                for box_file, mask_file in zip(boxes_order, instance_masks_order):
                    box_indice = box_file[0]
                    box_id = int(box_file[1].split(".")[0].split("_")[1])
                    mask_id = (int(mask_file[1].split(".")[0].split("_")[1]))
                    if pre_indice == None or box_indice == pre_indice:
                        pre_indice = box_indice
                        box_item.append((box_id, box_file[1]))
                        mask_item.append((mask_id, mask_file[1]))
                        continue
                    indice_list.append(int(pre_indice))
                    boxes.append((pre_indice, box_item))
                    instance_mask.append((pre_indice, mask_item))
                    box_item = []
                    mask_item = []
                    pre_indice = box_indice
                    box_item.append((box_id, box_file[1]))
                    mask_item.append((mask_id, mask_file[1]))
                indice_list.append(int(pre_indice))
                boxes.append((pre_indice, box_item))
                instance_mask.append((pre_indice, mask_item))
                #按帧处理GT_TRACK中的坐标
                pre_indice=None
                for coord_file in coords_order:
                    coord_indice, coord_id = map(int, coord_file[0].split("_"))
                    if pre_indice == None or coord_indice == pre_indice:
                        pre_indice = coord_indice
                        coord_item.append((coord_id, coord_file[1]))
                        continue
                    coords.append((pre_indice,coord_item))
                    coord_item = []
                    pre_indice = coord_indice
                    coord_item.append((coord_id, coord_file[1]))
                coords.append((pre_indice, coord_item))
                images.sort(key=lambda x: x[0])
                boxes.sort(key=lambda x: x[0])
                instance_mask.sort(key=lambda x: x[0])
                coords.sort(key=lambda x: x[0])
                length = len(images)
                non = 0
                total_i =0
                gt_seg_length = len(boxes)
                for i in range(gt_seg_length):
                    indice = indice_list[i]
                    #首张图片
                    track_intervals = 6
                    image_path = path + dir + "/images/" + images[indice][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    box_set = boxes[i][1]
                    instance_mask_set = instance_mask[i][1]
                    box_set.sort(key=lambda x:x[0])
                    instance_mask_set.sort(key=lambda x:x[0])
                    instance_num = len(instance_mask_set)  
                    for ii in range(max(0, indice-track_intervals), min(indice+track_intervals, length)):     
                        for j in range(instance_num):
                            self.annotations.append(self.get_annotations_track(box_set[j], total_i + num, annid, 1, instance_mask_set[j], path + dir, first=True))
                            annid += 1    
                        #后续图片
                        image_path_second = path + dir + "/images/" + images[ii][1]
                        instance_mask_set_second = coords[ii][1]
                        instance_mask_set_second.sort(key=lambda x:x[0])
                        instance_num_second = len(instance_mask_set_second) 
                        for jj in range(instance_num_second):
                            self.annotations.append(self.get_annotations_track_simi(total_i + num, annid, 1, instance_mask_set_second[jj], path + dir, first=False))
                            annid += 1 
                        #处理分裂事件 Fluo-N2DH-GOWT1数据集看不出分裂固在训练时省略该步骤
                        moists = {}
                        # for m in lineage:
                        #     m = m.strip().split(" ")
                        #     if int(m[3]) > 0:
                        #         if i < int(m[1]) <= ii:
                        #             if int(m[3]) not in moists:
                        #                 moists[int(m[3])] = []
                        #             moists[int(m[3])].append(int(m[0]))
                        self.images.append(self.get_images_track(image_path, image_path_second, h, w, total_i + num, moists))
                        total_i += 1 
                num = num + total_i + 1 - non
        self.coco["images"] = self.images
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations

    def test_to_coco(self):
        annid = 0
        sub_dirs = ["01","02"]
        num = 0
        for path in self.data_path:
            for dir in sub_dirs:
                images = [
                    (int(image_file.split(".")[0][1:]), image_file, )
                    for image_file in os.listdir(path + dir)
                    ]
                boxes = []
                instance_mask = []
                masks = []
                images.sort(key=lambda x: x[0])
                masks.sort(key=lambda x: x[0])
                boxes.sort(key=lambda x: x[0])
                instance_mask.sort(key=lambda x: x[0])
                length = len(images)
                for i in range(length):
                    image_path = path + dir + "/" + images[i][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    #self.annotations.append(self.get_annotations_test( i + num, annid, 1))          
                    self.images.append(self.get_images(image_path, h, w, i + num))
                num = num + i + 1
        self.coco["images"] = self.images
        #self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        #self.categories.append(self.get_categories("duplicate_box", self.class_ids["duplicate_box"]))
        self.categories.append(self.get_categories("background", self.class_ids["background"]))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations
        # print(self.coco)

    def get_images(self, filename, height, width, image_id):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        # 文件名加后缀
        image["file_name"] = filename

        #加入后续图片

        # print(image)
        return image

    def get_images_track(self, filename, filename_second, height, width, image_id, moists):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        # 文件名加后缀
        image["file_name"] = filename
        image["moists"] = moists
        #加入后续图片
        image["file_name_second"] = filename_second
        # print(image)
        return image
 
    def get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "Positive Cell"
        # id=0
        category['id'] = class_id
        # name=1
        category['name'] = name
        # print(category)
        return category
    
    def get_annotations_test(self, image_id, ann_id, cls):
        box = []
        annotation = {}
        #box = convert(box)
        annotation['segmentation'] = None
        annotation["instance_segmentation"] = None
        annotation['iscrowd'] = 0
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = 0
        # category_id=0
        annotation['category_id'] = cls
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        # 目标的id
        annotation["object_id"] = 0
        return annotation

    def get_annotations(self, box1, image_id, ann_id, cls, mask_file, instance_mask, path):
        box = torch.load(path + "/bounding_boxes/" +  box1[1])
        annotation = {}
        object_id = int(box[0])
        box = convert(box)
        box = rbb2hbb(box)
        #box = convert2xywh(box)
        w, h = box[2], box[3]
        area = w * h
        annotation['segmentation'] =  path + "/masks/" + mask_file
        annotation["instance_segmentation"] = path + "/instance_mask/" +instance_mask[1]
        annotation['iscrowd'] = 0
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = float(area)
        # category_id=0
        annotation['category_id'] = cls
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        # 目标的id
        annotation["object_id"] = object_id
        return annotation


    def get_annotations_track(self, box1, image_id, ann_id, cls, instance_mask, path, first=False):
        box = torch.load(path + "/bounding_boxes/" +  box1[1])
        object_id = int(box[0])
        annotation = {}
        #object_id = int(box[0])
        #box = convert2xywh(box)
        #annotation['segmentation'] =  path + "/masks/" + mask_file
        isntance_segmentation = path + "/instance_mask/" + instance_mask[1]
        segm = torch.load(isntance_segmentation)
        segm[segm>0] = 1
        contours, _ = cv2.findContours(segm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 0:
        #         contours = contours[0].astype(np.float32) #(points,1,2)
        #         r = cv2.minAreaRect(contours)
        #         obbox = np.array([r[0][0], r[0][1], r[1][1], r[1][0], 90 - r[2]])   
        # if box[1] - obbox[0] > 10:
        #     print(instance_mask)
        #     print(box1)
        polygons = [list(map(int, contours[0].reshape(-1)))]
        annotation['segmentation'] = polygons
        annotation["first"] = first
        annotation['iscrowd'] = 0
        # annotation['bbox'] = list(box[1:])
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        # category_id=0
        annotation['category_id'] = cls
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        # 目标的id
        annotation["object_id"] = object_id
        return annotation
 
    def get_annotations_track_simi(self, image_id, ann_id, cls, instance_mask, path, first=False):
        object_id = int(instance_mask[0])
        annotation = {}
        #object_id = int(box[0])
        #box = convert2xywh(box)
        #annotation['segmentation'] =  path + "/masks/" + mask_file
        isntance_segmentation = path + "/coords/" + instance_mask[1]
        # 初始化一个空列表来存储坐标
        coordinates = []
        # 打开文件并读取每一行
        with open(isntance_segmentation, 'r') as file:
            for line in file:
            # 去除行末的换行符，并按空格分割坐标
                x, y = line.strip().split()
                # 将坐标转换为浮点数并添加到列表中
                coordinates.append((float(x), float(y)))
        coordinates = np.array(coordinates)
        annotation['segmentation'] = [list(map(int, coordinates.reshape(-1)))]
        annotation["first"] = first
        annotation['iscrowd'] = 0
        # annotation['bbox'] = list(box[1:])
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        # category_id=0
        annotation['category_id'] = cls
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        # 目标的id
        annotation["object_id"] = object_id
        return annotation

    def save_json(self):
        self.npz_to_coco_val()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances_hbb.json'), 'w+')
        f.write(instances_train2017)
        f.close()
 
    def save_json_test(self):
        self.test_to_coco()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances.json'), 'w+')
        f.write(instances_train2017)
        f.close()

    def save_json_train(self):
        self.npz_to_coco_train()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances.json'), 'w+')
        f.write(instances_train2017)
        f.close()

    def save_json_track(self):
        self.npz_to_coco_train()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances_track.json'), 'w+')
        f.write(instances_train2017)
        f.close()

    def save_json_semi(self):
        self.npz_to_coco_simi()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances_track.json'), 'w+')
        f.write(instances_train2017)
        f.close()
    
    def save_json_train_semi(self):
        self.npz_to_coco_train_simi()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances_track.json'), 'w+')
        f.write(instances_train2017)
        f.close()
# 可改为train2017，要对应上面的
#paths = ['datasets/crops/DIC-C2DH-HeLa/train/', 'datasets/crops/Fluo-N2DH-GOWT1/train/', 'datasets/crops/Fluo-N2DL-HeLa/train/', 'datasets/crops/PhC-C2DH-U373/train/']
#paths = ['datasets/Fluo-N2DH-GOWT1/train/']
#paths = ['datasets/Fluo-N2DH-SIM+/train/']
paths = ['datasets/deepcell/train/']
#paths = ['datasets/Fluo-C2DL-MSC/train/']
#paths = ['datasets/Fluo-N2DH-GOWT1-test/']
#paths = ['datasets/Fluo-N2DL-HeLa-test/']
#paths = ['datasets/DIC-C2DH-HeLa/val/', 'datasets/Fluo-N2DH-GOWT1/val/', 'datasets/Fluo-N2DL-HeLa/val/', 'datasets/PhC-C2DH-U373/val/']
#paths = ['datasets/Fluo-C2DL-Huh7/']
# 保存地址
#save_path = 'datasets/Fluo-C2DL-Huh7/'
#save_path = 'datasets/crops/'
#save_path = 'datasets/Fluo-C2DL-MSC/train/'
#save_path = 'datasets/Fluo-N2DH-GOWT1-test/01/'
#save_path = 'datasets/Fluo-N2DL-HeLa-test/02/'
#save_path = 'datasets/Fluo-N2DH-GOWT1/train/'
save_path = 'datasets/deepcell/train/'
c = tococo(paths, save_path)
c.save_json()
#c.save_json_track()
#c.save_json_test()
#c.save_json_train_semi()