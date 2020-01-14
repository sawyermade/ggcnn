import os
import glob
import numpy as np

from .grasp_data_dir import GraspDatasetBaseDir
from utils.dataset_processing import grasp, image


class RsDirDataset(GraspDatasetBaseDir):
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
        super(RsDirDataset, self).__init__(**kwargs)

        # graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        graspf = glob.glob(os.path.join(file_path, '*-depth_vis.png'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('depth_vis', 'depth') for f in graspf]
        rgbf = [f.replace('depth', 'color') for f in depthf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def _get_crop_attrs(self, idx):
        # gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = [240, 320]
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

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_png(self.depth_files[idx])
        # print('depth_img.img 1', depth_img.img.shape)
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # print('depth_img.img 2', depth_img.img.shape)
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        # print('rgb_img.img 1', rgb_img.img.shape)
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # print('rgb_img.img', rgb_img.img.shape)
        # print('rgb_img.img 2', rgb_img.img.shape)
        return rgb_img.img
