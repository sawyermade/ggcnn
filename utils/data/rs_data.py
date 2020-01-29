import os
import glob
import numpy as np

from .grasp_data_rs import GraspDatasetBaseDir, GraspRsDataset
from utils.dataset_processing import grasp, image

import cv2, jsonpickle, requests

class RsDataset(GraspRsDataset):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(RsDataset, self).__init__(**kwargs)

        self.file_path = file_path

        # graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        graspf = glob.glob(os.path.join(file_path, '*rs-depth.png'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        # depthf = [f.replace('depth_vis', 'depth') for f in graspf]
        depthf = graspf
        rgbf = [f.replace('depth', 'rgb') for f in depthf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def _get_crop_attrs(self, idx, center=[240, 320]):
        # gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        # center = [240, 320]
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        grs = []
        gr = np.array([(0,0), (640,0), (640,480), (0,480)])
        grs.append(grasp.GraspRectangle(gr))
        gtbbs = grasp.GraspRectangles(grs)
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0, center_list=None):
        depth_img = image.DepthImage.from_png(self.depth_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # print('depth_img.img 1', depth_img.img.shape)
        if center_list == None:
            center, left, top = self._get_crop_attrs(idx)
        else:
            # print(f'\nIN CENTER_LIST:\n{center_list}\n')
            center, left, top, mask = center_list
            depth_img.img = (depth_img.img * mask[:, :, 0]) / 1000.0
            # depth_img.img = depth_img.img / 1000.0

        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # print('depth_img.img 2', depth_img.img.shape)

        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        ret_list = None
        # color_img = cv2.imread(self.rgb_files[idx], -1)
        # temp_idx = idx
        color_img = cv2.imread(self.rgb_files[idx], -1)
        # print(f'\n**** IDX = {idx} ****\n')
        flag_detectron = True
        while ret_list == None or flag_detectron:
            ret_list = self.upload(color_img)
        
            
            object_name = 'remote'
            mask_list = ret_list[-1]
            label_list = ret_list[2]
            bb_list = ret_list[1]
            print('Labels from Detectron: \n', label_list)
            # object_name = input('Enter object: ')
            
            if object_name in label_list:
                which_mask = label_list.index(object_name)
                flag_detectron = False

        # object_name = 'remote'
        # mask_list = ret_list[-1]
        # label_list = ret_list[2]
        # bb_list = ret_list[1]
        # which_mask = label_list.index(object_name)
        mask_temp = mask_list[which_mask]
        bb_temp = bb_list[which_mask]
        mask = np.zeros(color_img.shape, dtype=np.uint8)
        mask[:,:,0] = np.copy(mask_temp)
        mask[:,:,1] = np.copy(mask_temp)
        mask[:,:,2] = np.copy(mask_temp)
        mask_white = np.copy(mask)
        mask_white[mask_white < 1] = 255
        mask_white[mask_white == 1] = 0
        color_img_masked = mask * color_img + mask_white
        # fname = 'temp_rgb_masked.png'
        # temp_out_path = os.path.join(self.file_path, fname)
        # cv2.imwrite(temp_out_path, color_img_masked)

        bb_dims = [bb_temp[2] - bb_temp[0], bb_temp[3] - bb_temp[1]]
        center = [int(bb_temp[1] + bb_dims[1] // 2), int(bb_temp[0] + bb_dims[0] // 2)]

        # rgb_img = image.Image.from_file(temp_out_path)
        rgb_img = image.Image(cv2.cvtColor(color_img_masked, cv2.COLOR_BGR2RGB))
        # rgb_img = image.Image.from_file(self.rgb_files[idx])
        # print('rgb_img.img 1', rgb_img.img.shape)
        center, left, top = self._get_crop_attrs(idx, center)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # print('rgb_img.img', rgb_img.img.shape)
        # print('rgb_img.img 2', rgb_img.img.shape)
        return rgb_img.img, (center, left, top, mask)

    # Uploads to Detectron
    def upload(self, frame, url='http://127.0.0.1:665'):
        # Prep headers for http req
        content_type = 'application/json'
        headers = {'content_type': content_type}

        # jsonpickle the numpy frame
        _, frame_png = cv2.imencode('.png', frame)
        frame_json = jsonpickle.encode(frame_png)

        # Post and get response
        try:
            response = requests.post(url, data=frame_json, headers=headers)
            if response.text:
                # Decode response and return it
                retList = jsonpickle.decode(response.text)
                retList[0] = cv2.imdecode(retList[0], cv2.IMREAD_COLOR)
                retList[-1] = [cv2.imdecode(m, cv2.IMREAD_GRAYSCALE) for m in retList[-1]]
                
                # returns [vis.png, bbList, labelList, scoreList, maskList]
                return retList
            else:
                return None
        except:
            return None