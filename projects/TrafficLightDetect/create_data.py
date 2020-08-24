import sys
import yaml
import numpy as np
import os, json, cv2, random
import shutil

def judge_create_dir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    '''
    else:
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    '''

def intersection(box1, box2):

    inter_box_xmin = max(box1[0],box2[0])
    inter_box_ymin = max(box1[1],box2[1])
    inter_box_xmax = min(box1[2],box2[2])
    inter_box_ymax = min(box1[3],box2[3])
    inter_box = [inter_box_xmin,inter_box_ymin,inter_box_xmax,inter_box_ymax]
    w = max(inter_box[2]-inter_box[0],0)
    h = max(inter_box[3]-inter_box[1],0)
    inter = w * h 
    return inter_box,inter


#occluded annotation full light or part?
def get_crop_box(img_dir, img_ann, crop_scale, min_crop_size, iou_threshold, size_threshold=100):

    crop_img_anns_list = []
    
    filename = img_ann["path"]
    annos = img_ann["boxes"]
    abs_filename = os.path.join(img_dir,filename)

    try:
        img = cv2.imread(abs_filename)
        height, width = img.shape[:2]
    except:
        print("{} not exist".format(abs_filename))
        return crop_img_anns_list

    for i,anno in enumerate(annos):

        crop_img_anns = {}
        crop_objs = []
        
        fname,ext = os.path.splitext(filename)

        crop_filename = './crop_train_imgs/' + fname + "_%d"%i + ext
        print(crop_filename)
        
        xl = max(round(anno['x_min']),0)
        yt = max(round(anno['y_min']),0)
        xr = min(round(anno['x_max']),width)
        yb = min(round(anno['y_max']),height)

        roi_width = xr - xl + 1
        roi_height = yb - yt + 1

        #how to limit triffic light size
        if max(roi_width,roi_height) < size_threshold:
            continue

        center_x = int((xr + xl) / 2)
        center_y = int((yb + yt) / 2)
        
        #crop resize*resize square
        resize = int(crop_scale * max(roi_width,roi_height))
        resize = max(resize, min_crop_size)
        resize = min(resize, width)
        resize = min(resize, height)

        crop_xl = int(center_x - resize / 2 + 1)
        crop_xl = 0 if crop_xl < 0 else crop_xl
        crop_yt = int(center_y - resize / 2 + 1)
        crop_yt = 0 if crop_yt < 0 else crop_yt
        crop_xr = crop_xl + resize - 1
        crop_yb = crop_yt + resize - 1
        if crop_xr >= width - 1:
            crop_xl -= crop_xr - width + 1
            crop_xr = width - 1

        if crop_yb >= height - 1:
            crop_yt -= crop_yb - height + 1
            crop_yb = height - 1
        
        crop_box = [crop_xl,crop_yt,crop_xr,crop_yb]

        crop_img = img[crop_yt:crop_yb+1,crop_xl:crop_xr+1]
        abs_crop_filename = os.path.join(img_dir,crop_filename)
        judge_create_dir(os.path.split(abs_crop_filename)[0])

        cv2.imwrite(abs_crop_filename,crop_img)

        x_min = xl - crop_xl
        y_min = yt - crop_yt
        x_max = x_min + roi_width - 1
        y_max = y_min + roi_height - 1
        obj = {
            'label': anno['label'],
            "occluded": anno['occluded'],
            "centric": True,
            'x_max': x_max,
            'x_min': x_min,
            'y_max': y_max,
            'y_min': y_min, 
        }
        crop_objs.append(obj)

        for j in range(len(annos)):
            if j != i:
                other_box = [round(annos[j]['x_min']),round(annos[j]['y_min']),
                             round(annos[j]['x_max']),round(annos[j]['y_max'])]
                inter_box,inter = intersection(crop_box,other_box)
                other_box_areas = (other_box[2]-other_box[0]+1)*(other_box[3]-other_box[1]+1)
                
                if inter/other_box_areas > iou_threshold:

                    new_inter_box_w = inter_box[2] - inter_box[0] + 1
                    new_inter_box_h = inter_box[3] - inter_box[1] + 1
                    if max(new_inter_box_w,new_inter_box_h) < size_threshold/2:
                        continue

                    new_inter_box_xmin = inter_box[0] - crop_xl
                    new_inter_box_ymin = inter_box[1] - crop_yt
                    new_inter_box_xmax = new_inter_box_xmin + (inter_box[2] - inter_box[0])
                    new_inter_box_ymax = new_inter_box_ymin + (inter_box[3] - inter_box[1])
                    obj = {
                        'label': annos[j]['label'],
                        'occluded': annos[j]['occluded'],
                        "centric": False,
                        'x_max': new_inter_box_xmax,
                        'x_min': new_inter_box_xmin,
                        'y_max': new_inter_box_ymax,
                        'y_min': new_inter_box_ymin, 
                    }
                    crop_objs.append(obj)

        crop_img_anns['boxes'] = crop_objs
        crop_img_anns['path'] = crop_filename

        crop_img_anns_list.append(crop_img_anns)
    
    return crop_img_anns_list


def traffic_lights_preprocess(yaml_path,img_dir):
    crop_scale = 2.5
    min_crop_size = 270
    iou_threshold = 0.5
    size_threshold = 50

    imgs_anns = yaml.load(open(yaml_path, 'rb'))

    all_img_anns = []

    num = 0
    
    for img_ann in imgs_anns:
        crop_img_anns_list = get_crop_box(img_dir,img_ann,crop_scale,min_crop_size,iou_threshold,size_threshold)
        if len(crop_img_anns_list) > 0 :
            all_img_anns.extend(crop_img_anns_list)
            num = num + len(crop_img_anns_list)

    print(num)
    
    with open('./new_train.yaml','w') as f:
        yaml.dump(all_img_anns,f)
    

if __name__ == "__main__":
    #BSTLD
    #yaml_path = "/data/traffic_lights_data/train.yaml"
    #img_dir = "/data/traffic_lights_data/"
    #LISA
    yaml_path = "/data/traffic_lights_data/LISA/Yaml_Annotations/LISA_all.yml"
    img_dir = "/data/traffic_lights_data/"
    traffic_lights_preprocess(yaml_path,img_dir)












        

