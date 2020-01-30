import argparse
import logging

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

import os, cv2
import pyrealsense2 as rs
import numpy as np, requests, jsonpickle

logging.basicConfig(level=logging.INFO)

# Uploads to Detectron
def upload(frame, url='http://127.0.0.1:665'):
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

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    # Device
    parser.add_argument('--cuda', type=int, default=0, help='Cuda device number')


    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    # Get args
    args = parse_args()

    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    # Create temp_img_dir if it doesnt exist
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    # Realsense Setup
    width_rs, height_rs, fps_rs = 640, 480, 60
    pipeline_rs = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.depth, width_rs, height_rs, rs.format.z16, fps_rs)
    config_rs.enable_stream(rs.stream.color, width_rs, height_rs, rs.format.bgr8, fps_rs)
    profile_rs = pipeline_rs.start(config_rs)

    # Realsense alignment
    align_to = rs.stream.color
    align = rs.align(align_to)

    # First few images are no good until exposure adjusts
    count = 0
    while count < 60:
        # Get frames
        frames = pipeline_rs.wait_for_frames()
        count += 1

    # # Load Dataset
    # logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    # Dataset = get_dataset(args.dataset)
    # test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
    #                        random_rotate=args.augment, random_zoom=args.augment,
    #                        include_depth=args.use_depth, include_rgb=args.use_rgb)
    # test_data = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.num_workers
    # )
    # logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        flag_run = True
        while flag_run:
            # Get realsense images
            frames = pipeline_rs.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            # Gets Image masks and object of interest
            flag_detectron = True
            ret_list = None
            while ret_list == None or flag_detectron:
                ret_list = upload(color_img)
                if ret_list == None:
                    continue
            
                mask_list = ret_list[-1]
                label_list = ret_list[2]
                bb_list = ret_list[1]
                print('Objects Found:')
                for ln, label in enumerate(label_list):
                    print(f'{ln+1}. {label}')

                object_idx = int(input('Enter Object Number, zero to refresh: ')) - 1
                if object_idx < 0 or object_idx >= len(label_list):
                    print('Object Not Found. Refreshing Object Detection...\n')
                else:
                    flag_detectron = False

            # Applies mask to rgb and depth imgs
            mask_temp = mask_list[object_idx]
            bb_temp = bb_list[object_idx]
            mask = np.zeros(color_img.shape, dtype=np.uint8)
            mask[:,:,0] = np.copy(mask_temp)
            mask[:,:,1] = np.copy(mask_temp)
            mask[:,:,2] = np.copy(mask_temp)
            mask_white = np.copy(mask)
            mask_white[mask_white < 1] = 255
            mask_white[mask_white == 1] = 0
            color_img = mask * color_img + mask_white
            depth_img = depth_img * mask[:,:,0]

            # Gets center from bounding box
            bb_dims = [bb_temp[2] - bb_temp[0], bb_temp[3] - bb_temp[1]]
            center_pt = [int(bb_temp[1] + bb_dims[1] // 2), int(bb_temp[0] + bb_dims[0] // 2)]
                
            # Saves frames
            temp_img_dir = args.dataset_path
            temp_img_color_fname = 'temp_img_rs-rgb.png'
            temp_img_depth_fname = 'temp_img_rs-depth.png'
            cv2.imwrite(os.path.join(temp_img_dir, temp_img_color_fname), color_img)
            cv2.imwrite(os.path.join(temp_img_dir, temp_img_depth_fname), depth_img)

            # Load Dataset
            logging.info('Loading {} Dataset...'.format(args.dataset.title()))
            Dataset = get_dataset(args.dataset)
            test_dataset = Dataset(args.dataset_path, center_pt, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                                   random_rotate=args.augment, random_zoom=args.augment,
                                   include_depth=args.use_depth, include_rgb=args.use_rgb)
            test_data = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )
            logging.info('Done')

            for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
                # print('x size 3', x.size())
                logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc)


                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                       no_grasps=args.n_grasps,
                                                       grasp_width=width_img,
                                                       )
                    if s:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1

                if args.jacquard_output:
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                    with open(jo_fn, 'a') as f:
                        for g in grasps:
                            f.write(test_data.dataset.get_jname(didx) + '\n')
                            f.write(g.to_jacquard(scale=1024 / 300) + '\n')

                if args.vis:
                    rgb_gotten = test_data.dataset.get_rgb(didx, rot, zoom, normalise=False)
                    depth_gotten = test_data.dataset.get_depth(didx, rot, zoom)
                    evaluation.plot_output(rgb_gotten, depth_gotten, q_img,
                                           ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))