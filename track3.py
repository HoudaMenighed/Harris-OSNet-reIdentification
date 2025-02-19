import argparse
import time
import os
from tkinter import Tk, filedialog
import subprocess

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pygame


def load_selected_options():
    try:
        with open("selected_options.txt", "r") as file:
            selected_option = file.readline().strip()
            selected_option1 = file.readline().strip()
            selected_option2 = file.readline().strip()
            selected_option3 = file.readline().strip()
            return selected_option, selected_option1, selected_option2, selected_option3
    except FileNotFoundError:
        return None, None, None, None

selected_option, selected_option1,selected_option2,selected_option3= load_selected_options()
print(f"Selected Option: {selected_option}")
print(f"Selected Option 1: {selected_option1}")
print(f"Selected Option 2: {selected_option2}")
print(f"Selected Option 3: {selected_option3}")



def get_font(size):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    font_path = os.path.join(script_dir, "Night Monday.otf")

    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        print(f"Cannot find font file: {font_path}")
        raise


def get_font2(size):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    font_path = os.path.join(script_dir, "BEQINER.otf")

    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        print(f"Cannot find font file: {font_path}")
        raise


def get_font3(size):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    font_path = os.path.join(script_dir, "OpenSans-Light.ttf")

    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        print(f"Cannot find font file: {font_path}")
        raise

def open_video_in_default_player(video_path):
    try:
        if sys.platform.startswith('darwin'):
            subprocess.run(['open', video_path])
        elif sys.platform.startswith('linux'):
            subprocess.run(['xdg-open', video_path])
        elif sys.platform.startswith('win32'):
            os.startfile(video_path)
    except Exception as e:
        print(f"Failed to open video: {e}")

def select_video():
    Tk().withdraw()  # Prevents the root window from appearing
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    return video_path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'tracker') not in sys.path:
    sys.path.append(str(ROOT / 'tracker'))  # add tracker ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from tracker.utils.parser import get_config
from tracker.main_track import TrackerMain


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


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

    if show_vid:
        pygame.init()
        screen = pygame.display.set_mode((1220, 640))
        icon = pygame.image.load('track.jpg')
        pygame.display.set_icon(icon)

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

    # Load model
    device = select_device(device)

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset.sources)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

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


    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    start_time = time.time()

    tracked_object_ids = set()
    trajectory = {}

    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        frame_start_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)

        dt[2] += time_synchronized() - t3


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name) + str(i)  # im.jpg, vid.mp4, ...

            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop


            trackerList[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
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

                current_tracked_ids = sorted([int(track[4]) for track in outputs[i]])

                # Format each ID with 'p' prefix and join them with comma
                current_tracked_ids1 = (f"p{id}" for id in current_tracked_ids)

                # Print the formatted string
                print(current_tracked_ids1)

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]


                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g,' * 9 + '%g\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = (f'{id}')

                            """label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))"""

                            if selected_option == "RED":
                                selected_color = (0, 0, 255)  # Red color
                            elif selected_option == "GREEN":
                                selected_color = (0, 255, 0)  # Green color
                            elif selected_option == "BLUE":
                                selected_color = (255, 0, 0)  # Blue color
                            else:
                                selected_color = (
                                255, 0, 0)  # Default to white color if selected_option is not valid
                            print(f"Selected Color: {selected_color}")


                            plot_one_box(bboxes, im0, label=label, color=selected_color, line_thickness=4)

                        if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                        if selected_option3=="Show":
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
                                    pass

                        tracked_object_ids.add(f"p{int(id)}")




                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), TrackerMain:({t5 - t4:.3f}s)')

            else:

                trackerList[i].increment_ages()
                print('No detections')


            if show_vid:
                display_frame = cv2.resize(im0, (810, 480))

                blank_frame = np.zeros((480, 810, 3), dtype=np.uint8)

                y_offset = (blank_frame.shape[0] - display_frame.shape[0]) // 2
                x_offset = (blank_frame.shape[1] - display_frame.shape[1]) // 2

                blank_frame[y_offset:y_offset + display_frame.shape[0],
                x_offset:x_offset + display_frame.shape[1]] = display_frame

                blank_frame_rgb = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB)

                im0_pygame = pygame.image.frombuffer(blank_frame_rgb.tobytes(), blank_frame_rgb.shape[1::-1], 'RGB')


                if screen is None:
                    screen = pygame.display.set_mode((blank_frame_rgb.shape[1], blank_frame_rgb.shape[0]))
                script_dir27 = os.path.dirname(os.path.abspath(__file__))
                image_path27 = os.path.join(script_dir27, "b10.jpg")
                BG7 = pygame.image.load(image_path27)
                BG7 = pygame.transform.scale(BG7, (1220, 640))
                screen.blit(BG7, (0, 0))

                script_dir29 = os.path.dirname(os.path.abspath(__file__))
                image_path29 = os.path.join(script_dir29, "exit2.png")
                image20 = pygame.image.load(image_path29)
                resized_image20 = pygame.transform.scale(image20, (35, 25))
                image_position20 = (21, 17)

                text_font = get_font2(35)
                text = text_font.render("Multi-Object Tracking", True, (200, 252, 192))
                screen.blit(text, (328, 8))

                pygame.draw.rect(screen, (255, 23, 38), (316, 57, 110, 4))
                pygame.draw.rect(screen, (255, 23, 38), (316, 20, 4, 40))

                pygame.draw.rect(screen, (255, 23, 38), (695, 8, 110, 4))
                pygame.draw.rect(screen, (255, 23, 38), (805, 8, 4, 40))

                pygame.draw.rect(screen, (174, 227, 163), (10, 85, 826, 4))
                pygame.draw.rect(screen, (174, 227, 163), (10, 85, 4, 498))
                pygame.draw.rect(screen, (174, 227, 163), (10, 583, 826, 4))
                pygame.draw.rect(screen, (174, 227, 163), (836, 85, 4, 502))

                pygame.draw.rect(screen, (174, 227, 163), (862, 26, 324, 4))
                pygame.draw.rect(screen, (174, 227, 163), (862, 26, 4, 593))
                pygame.draw.rect(screen, (174, 227, 163), (862, 619, 328, 4))
                pygame.draw.rect(screen, (174, 227, 163), (1186, 26, 4, 594))

                text_font = get_font3(23)
                text = text_font.render("Model :", True, (255, 255, 255))
                screen.blit(text, (908, 42))

                pygame.draw.rect(screen, (237, 247, 252), (1040, 44, 101, 30))

                text_font = get_font3(29)
                text = text_font.render("Yolo", True, (184, 4, 13))
                screen.blit(text, (1063, 39))

                text_font = get_font3(23)
                text = text_font.render("Class :", True, (255, 255, 255))
                screen.blit(text, (908, 96))

                pygame.draw.rect(screen, (237, 247, 252), (1040, 98, 101, 30))

                text_font = get_font3(29)
                text = text_font.render("Person", True, (184, 4, 13))
                screen.blit(text, (1045, 92))

                pygame.draw.rect(screen, (230, 46, 64), (930, 168, 190, 3))


                text_font = get_font3(24)
                text = text_font.render("FPS :", True, (255, 255, 255))
                screen.blit(text, (936, 216))
                pygame.draw.rect(screen, (255, 251, 227), (1040, 216, 84, 33))

                pygame.draw.rect(screen, (255, 251, 227), (934, 286, 180, 45))
                text_font = get_font3(27)
                text = text_font.render("frame nº", True, (0, 0, 0))
                screen.blit(text, (944, 290))



                text_font = get_font3(24)
                text = text_font.render("List of Object", True, (255, 255, 255))
                screen.blit(text, (881, 390))

                text_font = get_font3(24)
                text = text_font.render("Tracked :", True, (255, 255, 255))
                screen.blit(text, (911, 427))

                pygame.draw.rect(screen, (255, 251, 227), (1044, 390, 120, 72))

                text_font = get_font3(24)
                text = text_font.render("Current", True, (255, 255, 255))
                screen.blit(text, (907, 518))

                text_font = get_font3(24)
                text = text_font.render("Object :", True, (255, 255, 255))
                screen.blit(text, (907, 555))

                pygame.draw.rect(screen, (255, 251, 227), (1044, 520, 120, 72))

                exit_button_font = get_font(55)
                exit_button_color = (233, 250, 227)
                exit_button_rect = pygame.Rect(15, 12, 46, 35)
                pygame.draw.rect(screen, exit_button_color, exit_button_rect)
                exit_text = exit_button_font.render("", True, (0, 0, 0))
                screen.blit(exit_text, (566, 592))

                screen.blit(resized_image20, image_position20)


                screen.blit(im0_pygame, (20, 95))
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if exit_button_rect.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()
            pygame.time.wait(10)

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

        frame_end_time = time.time()
        fps = 1 / (frame_end_time - frame_start_time)




        if show_vid:
            fps_text = f"{fps:.2f}"
            font = pygame.font.Font(None, 36)
            fps_surface = font.render(fps_text, True, (0, 0, 0))
            screen.blit(fps_surface, (1058, 221))

            text_font = get_font3(29)
            text = text_font.render(f"{frame_idx + 1}", True, (201, 42, 58))
            screen.blit(text, (1065, 289))

            def chunk_list(lst, chunk_size):
                for i in range(0, len(lst), chunk_size):
                    yield lst[i:i + chunk_size]

            tracked_object_ids_list = list(tracked_object_ids)
            tracked_object_ids_list.sort()

            id_chunks = list(chunk_list(tracked_object_ids_list, 3))

            y_position = 388
            text_font = get_font3(24)

            for chunk in id_chunks:
                ids_string = ",".join(map(str, chunk))
                text = text_font.render(ids_string, True, (0, 0, 0))
                screen.blit(text, (1055, y_position))
                y_position += 39

            def chunk_list1(lst, chunk_size):
                for i in range(0, len(lst), chunk_size):
                    yield lst[i:i + chunk_size]

            tracked_object_ids_list1 = list(current_tracked_ids1)
            tracked_object_ids_list1.sort()

            id_chunks = list(chunk_list1(tracked_object_ids_list1, 3))

            y_position = 517
            text_font = get_font3(24)

            for chunk in id_chunks:
                ids_string = ",".join(map(str, chunk))
                text = text_font.render(ids_string, True, (0, 0, 0))
                screen.blit(text, (1055, y_position))
                y_position += 39

            pygame.display.update()

    return tracked_object_ids,current_tracked_ids1


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    print('Done. (%.3fs)' % (time.time() - start_time))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--Tweights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
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
    if selected_option1 == 'Yes':
        opt.save_vid = True
        opt.save_txt=True
    elif selected_option1 == 'No':
        opt.save_vid = False

    if selected_option2 == 'Show':
        opt.hide_class = False
    elif selected_option2 == 'Not Show':
        opt.hide_class = True

    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
