import json
import glob
import cv2 as cv
import os
import torch
import numpy
import tifffile


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
            for dir in sub_dirs:
                images = [
                    (int(image_file.split(".")[0]), image_file, )
                    for image_file in os.listdir(path + dir + "/images")
                    ]
                masks = [
                    (int(mask_file.split(".")[0]), mask_file)
                    for mask_file in os.listdir(path + dir + "/masks")
                    ]
                boxes_order = [
                    (int(box_file.split(".")[0].split("_")[0]), box_file)
                    for box_file in os.listdir(path + dir + "/bounding_boxes")
                    ]
                instance_masks_order = [
                    (int(instance_mask_file.split(".")[0].split("_")[0]), instance_mask_file)
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
                        if box_set[j][0] != instance_mask_set[j][0]:
                            print("ok")
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
    
    def npz_to_coco_train_crop(self):
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
                instance_masks_order = [
                    (instance_mask_file.split(".")[0], instance_mask_file)
                    for instance_mask_file in os.listdir(path + dir + "/instance_mask")
                    ]
                
                instance_masks_order.sort(key=lambda x: x[0])
                instance_mask = {}
                for  mask_file in instance_masks_order:
                    mask_indice = mask_file[0][0:7]
                    # mask_id = (int(mask_file[1].split(".")[0].split("_")[1])
                    if mask_indice not in instance_mask:
                        instance_mask[mask_indice] = []
                    instance_mask[mask_indice].append(mask_file[1])

                images.sort(key=lambda x: x[0])
                masks.sort(key=lambda x: x[0])
                length = len(images)
                non = 0
                for i in range(length):
                    image_path = path + dir + "/images/" + images[i][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    mask = masks[i][1]
                    if images[i][0] not in instance_mask:
                        non += 1
                        continue
                    instance_mask_set = instance_mask[images[i][0]]
                    instance_num = len(instance_mask_set)
                    for j in range(instance_num):
                        self.annotations.append(self.get_annotations_train(i + num, annid, 1, mask, instance_mask_set[j], path + dir))
                        annid += 1           
                    self.images.append(self.get_images(image_path, h, w, i + num))
                num = num + i + 1 - non
        self.coco["images"] = self.images
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations

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
                instance_masks_order = [
                    (instance_mask_file.split(".")[0], instance_mask_file)
                    for instance_mask_file in os.listdir(path + dir + "/instance_mask")
                    ]
                boxes_order = [
                    (box_file.split(".")[0], box_file)
                    for box_file in os.listdir(path + dir + "/bounding_boxes")
                    ]
                boxes_order.sort(key=lambda x: x[0])
                instance_masks_order.sort(key=lambda x: x[0])
                instance_mask = {}
                boxes = {}
                for  box_file,mask_file in zip(boxes_order,instance_masks_order):
                    #mask_indice = mask_file[0][0:7]
                    #mask_indice = int(mask_file[1].split(".")[0].split("_")[0])
                    mask_indice = int(mask_file[1].split(".")[0].split("_")[0])
                    box_indice = int(box_file[1].split(".")[0].split("_")[0])
                    if box_indice not in boxes:
                        boxes[box_indice] = []
                    if mask_indice not in instance_mask:
                        instance_mask[mask_indice] = []
                    boxes[box_indice].append(box_file[1])
                    instance_mask[mask_indice].append(mask_file[1])

                images.sort(key=lambda x: x[0])
                masks.sort(key=lambda x: x[0])
                length = len(images)
                non = 0
                for i in range(length):
                    image_path = path + dir + "/images/" + images[i][1]
                    image = tifffile.imread(image_path)
                    h, w = image.shape
                    mask = masks[i][1]
                    if int(images[i][0]) not in instance_mask:
                        non += 1
                        continue
                    instance_mask_set = instance_mask[int(images[i][0])]
                    instance_num = len(instance_mask_set)
                    boxes_set = boxes[int(images[i][0])]
                    for j in range(instance_num):
                        self.annotations.append(self.get_annotations_train(boxes_set[j], i + num, annid, 1, mask, instance_mask_set[j], path + dir))
                        annid += 1           
                    self.images.append(self.get_images(image_path, h, w, i + num))
                num = num + i + 1 - non
        self.coco["images"] = self.images
        self.categories.append(self.get_categories("cell", self.class_ids["cell"]))
        self.categories.append(self.get_categories("background", self.class_ids['background']))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations


    def test_to_coco(self):
        annid = 0
        sub_dirs = ["02"]
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
                    self.images.append(self.get_images("../PhC-C2DH-U373/" + dir + "/" + images[i][1], h, w, i + num))         
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
        #box = convert(box)
        box = convert2xywh(box)
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


    def get_annotations_train(self, box_file, image_id, ann_id, cls, mask_file, instance_mask, path):
        annotation = {}
        box = torch.load(path + "/bounding_boxes/" +  box_file)
        object_id = int(box[0])
        box = convert(box)
        #box = convert2xywh(box)
        annotation['bbox'] = box
        w, h = box[2], box[3]
        area = w * h
        annotation['area'] = float(area)
        annotation['segmentation'] =  path + "/masks/" + mask_file
        annotation["instance_segmentation"] = path + "/instance_mask/" + instance_mask
        annotation['iscrowd'] = 0
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        # category_id=0
        annotation['category_id'] = cls
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        # 目标的id
        # annotation["object_id"] = object_id
        return annotation
 
    def save_json(self):
        self.npz_to_coco_val()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        
        f = open(os.path.join(self.save_path + 'instances.json'), 'w+')
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

    def save_json_train_crop(self):
        self.npz_to_coco_train_crop()
        label_dic = self.coco
        # print(label_dic)
        instances_train = json.dumps(label_dic)
        f = open(os.path.join(self.save_path + 'instances.json'), 'w+')
        f.write(instances_train)
        f.close()

# 可改为train2017，要对应上面的
#paths = ['datasets/crops/DIC-C2DH-HeLa/train/', 'datasets/crops/Fluo-N2DH-GOWT1/train/', 'datasets/crops/Fluo-N2DL-HeLa/train/', 'datasets/crops/PhC-C2DH-U373/train/']

#paths = ['datasets/Fluo-N2DH-SIM+/val/']
paths = ['datasets/Fluo-N2DH-SIM+-test/']

#paths = ['datasets/Fluo-N2DH-GOWT1-test/']
#paths = ['datasets/PhC-C2DH-U373-test/']
#paths = ['datasets/Fluo-N2DL-HeLa-test/']
#paths = ['datasets/DIC-C2DH-HeLa/val/', 'datasets/Fluo-N2DH-GOWT1/val/', 'datasets/Fluo-N2DL-HeLa/val/', 'datasets/PhC-C2DH-U373/val/']
# 保存地址
#save_path = 'datasets/crops/'
save_path = 'datasets/Fluo-N2DH-SIM+-test/'
#save_path = 'datasets/Fluo-N2DH-GOWT1/val/'
#save_path = 'datasets/Fluo-N2DH-SIM+/val/'
#save_path = 'datasets/Fluo-N2DH-GOWT1-test/02/'
#save_path = 'datasets/Fluo-N2DL-HeLa-test/02/'
#save_path = 'datasets/PhC-C2DH-U373-test/02/'
c = tococo(paths, save_path)
#c.save_json()
#c.save_json_train()
c.save_json_test()
