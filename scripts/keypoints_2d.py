"""Script for single-gpu/multi-gpu demo."""
import os
import sys
import time
import platform
from argparse import Namespace

import numpy as np
import torch
from tqdm import tqdm
import mediapipe as mp
import cv2
import natsort

sys.path.append("/users/axing2/data/axing2/hand-pose/AlphaPose_mp")
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap, get_affine_transform, im_to_torch
from alphapose.utils.vis import getTime
from alphapose.utils.writer import DataWriter
from alphapose.utils.bbox import (_box_to_center_scale, _center_scale_to_box)

# ------------------------------ Media Pipe Bounding Boxes ------------------------------ #
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
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
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

        # just appends the score for hand classification since Alpha Pose doesn't use
        #   bounding box scores for detection
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

# ------------------------------ Alpha Pose Helpers ------------------------------ #

def check_input():
    # for images
    if len(args.inputpath) or len(args.inputimg): 
        inputpath = args.inputpath
        inputimg = args.inputimg

        if len(inputpath) and inputpath != '/':
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
    if (args.save_img) and not args.vis_fast: # or args.save_video
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1

# ------------------------------ 2D Key Points ------------------------------ #
def keypoints_2d(ap_args, cfg, pose_model, pose_dataset):
    global args
    args = ap_args

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    mode, input_source = check_input()

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
    queueSize = args.qsize
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()

    data_len = len(input_source)
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
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
                m_inps = m_inps.to(args.device)
                datalen = m_inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    m_inps_j = m_inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        m_inps_j = torch.cat((m_inps_j, flip(m_inps_j)))
                    hm_j = pose_model(m_inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
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
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues
            writer.terminate()
            writer.clear_queues()
