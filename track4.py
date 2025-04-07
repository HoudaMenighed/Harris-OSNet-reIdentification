import argparse

import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch;
import torch.backends.cudnn as cudnn
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from tracker.utils.parser import get_config
from tracker.main_track import TrackerMain

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # include video suffixes

import cv2
import numpy as np

import time

start_time = time.time()

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov7.pt',
        Tweights=WEIGHTS / 'osnet_x0_25_msmt1760.pt',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        show_vid=False,
        show_vid_path="C:/Users/STS/PycharmProjects/INTERFACE",
        save_txt=True,
        save_conf=False,
        save_crop=False,
        save_vid=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/track',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        hide_class=False,
        half=False,
        dnn=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + Tweights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device = select_device(device)

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset.sources)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    trackerList = []
    for i in range(nr_sources):
        trackerList.append(
            TrackerMain(
                Tweights,
                device,
                half,
                cosine_threshold=0.2,
                iou_threshold=0.7,
                max_age=500,
                frames=3,
                nn_=500,
                lambda_=0.995 ,
                alpha_=0.9,
            )
        )
        trackerList[i].model.warmup()
    outputs = [None] * nr_sources
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    frame_count = 0
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1
        t3 = time_synchronized()
        dt[1] += t3 - t2
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False

        pred = model(im)
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        dt[2] += time_synchronized() - t3

        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop


            if det is not None and len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                t4 = time_synchronized()
                outputs[i] = trackerList[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                dt[3] += t5 - t4
                frame_count += 1

                elapsed_time = time.time() - start_time

                fps = frame_count / elapsed_time

                cv2.putText(im0, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        """if draw:
                            # object trajectory
                            center = ((int(bboxes[0]) + int(bboxes[2])) // 2, (int(bboxes[1]) + int(bboxes[3])) // 2)
                            if id not in trajectory:
                                trajectory[id] = []
                            trajectory[id].append(center)
                            for i1 in range(1, len(trajectory[id])):
                                if trajectory[id][i1 - 1] is None or trajectory[id][i1] is None:
                                    continue
                                # thickness = int(np.sqrt(1000/float(i1+10))*0.3)
                                thickness = 2
                                try:
                                    cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 255, 0), thickness)
                                except:
                                    pass"""

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            txt_file_path = txt_path + '.txt'
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g,' * 9 + '%g\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                  (f'{id}'))
                            plot_one_box(bboxes, im0, label=label, color=(0, 0, 255), line_thickness=3)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), Tracker:({t5 - t4:.3f}s)')

            else:
                trackerList[i].increment_ages()
                print('No detections')

            if show_vid:
                inf = (f'{s}Done. ({t2 - t1:.3f}s)')
                # cv2.putText(im0, str(inf), (30,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    break

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    print('save path : ', save_path)
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    # Save in WebM format using VP8 codec
                    save_path = str(Path(save_path).with_suffix('.webm'))  # force *.webm suffix
                    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec (or 'VP90' for VP9)
                    vid_writer[i] = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--Tweights', type=str, default=WEIGHTS / 'osnet_x0_25_350.pt')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
