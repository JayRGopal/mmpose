# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

# In case there's no screen on the device
import matplotlib
matplotlib.use('Agg')

# Verify
from deepface import DeepFace
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)
from utilsVerify import *

# Pose tracking imports
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      target_face_path,
                      visualizer=None,
                      show_interval=0,
                      face_conf_thresh=0.7,
                      verifyAll=False):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # Is this frame valid? It is not valid if:
    # 1 - no poses detected with sufficient confidence
    # 2 - one pose, but verified and wasn't the target face
    # 3 - multiple poses, but verified and none were the target face
    is_valid = True

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    
    # get non-preprocessed image
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    # verify via deepface if >1 confident face landmarks
    num_ppl = len(pose_results)
    final_num_ppl = num_ppl # final number of people after thresholding & verificaiton (to be updated below)
    if num_ppl >= 1 and not(args.debug):
        face_confidences = avg_face_conf_body_2d(pose_results)
        num_people_above_threshold = np.sum(face_confidences >= face_conf_thresh)
        above_threshold_indices = np.where(face_confidences >= face_conf_thresh)[0]
        if (num_people_above_threshold > 1) or verifyAll:
            # Case: >1 confident face in frame
            extracted = [pose_results[i] for i in above_threshold_indices]
            result = verify_one_face_np_data(target_face_path, img)
            if result is None:
                final_num_ppl = 0
                data_samples = merge_data_samples([])
            else:
                face_x, face_y, face_w, face_h = result
                face_center_x = face_x + (face_w / 2)
                face_center_y = face_y + (face_h / 2) 
                if (num_people_above_threshold >= 1):
                    all_nose_coords = get_nose_coords_body_2d(extracted)
                    correct_person_index = closest_person_index(face_center_x, face_center_y, all_nose_coords)
                    if correct_person_index == -1:
                        final_num_ppl = 0
                        new_pose_results = []
                    else:
                        final_num_ppl = 1
                        new_pose_results = [extracted[correct_person_index]]
                else:
                    final_num_ppl = 0
                    new_pose_results = []
                data_samples = merge_data_samples(new_pose_results)
        elif num_people_above_threshold == 1:
            # Case: 1 confident face in frame 
            extracted = [pose_results[i] for i in above_threshold_indices]
            final_num_ppl = 1
            data_samples = merge_data_samples(extracted)
        elif num_people_above_threshold == 0: 
            # Case: no confident faces in frame
            final_num_ppl = 0
            data_samples = merge_data_samples([])

    else:
        data_samples = merge_data_samples(pose_results)
    
    if final_num_ppl == 0:
        # If we end up with no people, revert back to original preds for visualizer
        data_samples_vis = merge_data_samples(pose_results)

        # also, mark this frame as invalid
        is_valid = False
    else:
        data_samples_vis = data_samples
    
    # show the results
    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples_vis,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None), is_valid


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output json file. '
        'Default not saving the json with body keypoints.')
    parser.add_argument(
        '--output-video',
        type=str,
        default='',
        help='root of the output video file with keypoints overlayed. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='whether to show ALL poses for debugging purposes')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--target-face-path',
        type=str,
        default='',
        help='Path to the target image with out patient for verification') 
    parser.add_argument(
        '--verifyAll',
        action='store_true',
        default=False,
        help='Whether to verify every single frame (slower, but more robust)')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)

    if args.output_video:
       mmengine.mkdir_or_exist(args.output_video)
       output_file = os.path.join(args.output_video,
                                   os.path.basename(args.input))
       if args.input == 'webcam':
            output_file += '.mp4' 

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    
    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    
    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    
    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    
    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances, is_valid = process_one_image(args, args.input, detector,
                                           pose_estimator, args.target_face_path, visualizer=visualizer,
                                           verifyAll=args.verifyAll)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)
        
        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            # TODO: Add downsampling of frames

            

            pred_instances, is_valid = process_one_image(args, frame, detector,
                                               pose_estimator, args.target_face_path, 
                                               visualizer=visualizer,
                                               show_interval=0.001,
                                               verifyAll=args.verifyAll)

            
            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            if output_file:
                
                frame_vis = visualizer.get_image()
                
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        30,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))
                
                # Check if the frame is valid or not
                if not is_valid:
                    # Add "SKIPPED" text in bold red letters
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    top_left_corner_of_text = (10, 90)
                    font_scale = 2
                    font_color = (255, 0, 0)  # Red color in RGB
                    line_type = 3
                    cv2.putText(frame_vis, 'SKIPPED',
                                top_left_corner_of_text,
                                font,
                                font_scale,
                                font_color,
                                line_type)
                

                video_writer.write(mmcv.rgb2bgr(frame_vis))
                





            # press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

            time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        try:
            with open(args.pred_save_path, 'x') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
        except FileExistsError:
            with open(args.pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
        
        # with open(args.pred_save_path, 'w') as f:
        #     json.dump(
        #         dict(
        #             meta_info=pose_estimator.dataset_meta,
        #             instance_info=pred_instances_list),
        #         f,
        #         indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')


if __name__ == '__main__':
    main()

