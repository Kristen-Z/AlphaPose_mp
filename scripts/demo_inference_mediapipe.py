"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
sys.path.append("/users/axing2/data/axing2/hand-pose/AlphaPose_mp") # path to alpha pose directory
import time

import numpy as np
import torch
from tqdm import tqdm
import mediapipe as mp
import cv2
import natsort

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap, get_affine_transform, im_to_torch
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
from alphapose.utils.bbox import (_box_to_center_scale, _center_scale_to_box)

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

mp_hands = mp.solutions.hands
padding = 0.3
input_size = [256,192]
_aspect_ratio = float(input_size[1]) / input_size[0]

def test_transform(src, bbox):
    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin)
    scale = scale * 1.0

    inp_h, inp_w = input_size

    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)

    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    return img, bbox

def get_mediapipe_bbox(frame):
    def _get_output(boxes, scores):
        ids = torch.zeros((len(scores), len(scores[0])))
        inps = torch.zeros(len(boxes), 3, *input_size)
        cropped_boxes = torch.zeros(len(boxes), 4)
        if frame is None:
            return None, None, None, None, None
        if boxes is None:
            return None, boxes, None, scores, ids
        for i, box in enumerate(boxes):
            inps[i], cropped_box = test_transform(frame, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)

        return torch.Tensor(inps), torch.Tensor(boxes), torch.Tensor(cropped_boxes), torch.Tensor(scores), torch.Tensor(ids)

    bboxes = []
    scores = []
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        bbox = []
        image = cv2.flip(frame, 1)
        results = hands.process(image)

        image_height, image_width, _ = image.shape
        if not results.multi_handedness:
            img_sqr = [0, 0, image_width, image_height]
            if image_width < image_height:
                img_sqr[1] = image_width/4
                img_sqr[3] = image_width
            else:
                img_sqr[0] = image_height/4
                img_sqr[2] = image_height
            bboxes.append(img_sqr)
            scores.append([0.0])
            return _get_output(bboxes,scores)

        scores.append([results.multi_handedness[0].classification[0].score])
        if not results.multi_hand_landmarks:
            return torch.tensor([[0,0,0,0]])
        image_height, image_width, _ = image.shape
        landmarks = results.multi_hand_landmarks[0].landmark
        for landmark in landmarks:
            bbox.append([landmark.x * image_width, landmark.y * image_height])

        # Calculate bounding box
        bbox = np.array(bbox)
        bbox_min = bbox.min(0)
        bbox_max = bbox.max(0)
        bbox_size = bbox_max - bbox_min

        # Pad hand bounding box
        bbox_min -= bbox_size * padding
        bbox_max += bbox_size * padding
        bbox_size = bbox_max - bbox_min

        # Convert bbox to square of length equal
        # to longer edge
        diff = bbox_size[0] - bbox_size[1]
        if diff > 0:
            bbox_min[1] -= diff / 2
            bbox_max[1] += diff / 2
            bbox_size[1] = bbox_size[0]
        else:
            bbox_min[0] -= -diff / 2
            bbox_max[0] += -diff / 2
            bbox_size[0] = bbox_size[1]

        # Flip
        tmp = bbox_min[0]
        bbox_min[0] = image_width - bbox_max[0]
        bbox_max[0] = image_width - tmp
        image = cv2.flip(image, 1)

        bboxes.append([*bbox_min, *bbox_max])
        return _get_output(bboxes, scores)

def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":
    mode, input_source = check_input()

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    # if mode == 'webcam':
    #     det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
    #     det_worker = det_loader.start()
    # elif mode == 'detfile':
    #     det_loader = FileDetectionLoader(input_source, cfg, args)
    #     det_worker = det_loader.start()
    # else:
    #     det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
    #     det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    # if args.save_video and mode != 'image':
    #     from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
    #     if mode == 'video':
    #         video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
    #     else:
    #         video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
    #     video_save_opt.update(det_loader.videoinfo)
    #     writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
    # else:
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()

    # if mode == 'webcam':
    #     print('Starting webcam demo, press Ctrl + C to terminate...')
    #     sys.stdout.flush()
    #     im_names_desc = tqdm(loop())
    # else:
    data_len = len(input_source) # det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                # (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                im_name = input_source[i]
                orig_img = cv2.cvtColor(cv2.imread(os.path.join(args.inputpath, im_name)), cv2.COLOR_BGR2RGB)
                m_inps, m_boxes, m_cropped_boxes, m_scores, m_ids = get_mediapipe_bbox(orig_img)
                if orig_img is None:
                    break
                if m_boxes is None:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                # inps = inps.to(args.device)
                m_inps = m_inps.to(args.device)
                datalen = m_inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    # inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    m_inps_j = m_inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        # inps_j = torch.cat((inps_j, flip(inps_j)))
                        m_inps_j = torch.cat((m_inps_j, flip(m_inps_j)))
                    # hm_j = pose_model(inps_j)
                    hm_j = pose_model(m_inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                hm = hm.cpu()
                # writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                writer.save(m_boxes, m_scores, m_ids, hm, m_cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        # det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            # det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            # det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            # det_loader.clear_queues()


    

