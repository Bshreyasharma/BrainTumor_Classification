import json
import scipy.ndimage as ndi
import nibabel as nib
import numpy as np
from skimage.transform import resize
#---------------------------------------------------------------------------------------------------------------------------------
"""
Here the image is fetched, pre-processed for prediction
"""
def load_jsonFile():
    with open('dynamic/upload1/file_info.json', 'r') as openfile:

        # Reading from json file
        json_object = json.load(openfile)

    return json_object

file_info = load_jsonFile()

#---------------------------------------------------------------------------------------------------------------------------------
def find_max_tumor(image):
    #Takes in the numpy image and returns the layer index which contains the maximum size of tumor
    mask_sum=0
    layer_index=0
    for i in range(155): #as there are 155 height layers
        image_file=image[:,:,i]
        filt = ndi.gaussian_filter(image_file,sigma=1)
        mask = filt > 450
        labels, nlabels = ndi.label(mask)

        if i==154 and mask_sum==0:
            m=0
            l=0
            for j in range(155):
                image_file=image[:,:,j]
                filt= ndi.gaussian_filter(image_file,sigma=1)
                mask1 = filt > 250
                if mask1.sum() > m:
                    m=mask1.sum()
                    l=j
            mask_sum=m
            layer_index=l

        if mask.sum() > mask_sum:
            mask_sum=mask.sum()
            layer_index=i

    return layer_index

def pre_processImage():
    global file_info
    flair_image=str(file_info['file_path'])+str(file_info['flair'])
    flair_image = str(flair_image)
    flair_image = nib.load(flair_image)
    flair = np.array(flair_image.dataobj)

    t1 =str(file_info['file_path'])+str(file_info['t1'])
    t1 = str(t1)
    t1 = nib.load(t1)
    t1 = np.array(t1.dataobj)

    t1ce = str(file_info['file_path'])+str(file_info['t1ce'])
    t1ce = str(t1ce)
    t1ce = nib.load(t1ce)
    t1ce = np.array(t1ce.dataobj)

    t2 = str(file_info['file_path'])+str(file_info['t2'])
    t2 = str(t2)
    t2 = nib.load(t2)
    t2 = np.array(t2.dataobj)
    index = find_max_tumor(flair)

    res=np.concatenate((flair[:,:,index],t1[:,:,index],t1ce[:,:,index],t2[:,:,index]),axis=1)
    res = resize(res, (224, 224))
    res_image = []
    res_image.append(res)
    res_image = np.array(res_image)
    res_image = np.repeat(res_image[..., np.newaxis], 3, -1)
    return res_image

