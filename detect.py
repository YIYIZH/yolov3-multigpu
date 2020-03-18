import argparse
import time
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import json


def detect(save_txt=False, save_img=True, stream_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half = opt.output, opt.source, opt.weights, opt.half
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')

    # Initialize
    device = torch_utils.select_device(force_cpu=False)
    #device = torch.device("cuda:0")
    '''
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    '''
    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    # torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    if webcam:
        stream_img = True
        dataset = LoadWebcam(source, img_size=img_size, half=half)
    else:
        save_img = False
        save_txt = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes('data/zsd_unseen.names') # change model detect as well
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(140)]

    # Run inference
    t0 = time.time()
    save_img = True
    for path, img, im0, vid_cap in dataset:
        t = time.time()
        save_path = str(Path(out+ '/images') / Path(path).name)
        img_name = str(Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred.float(), opt.conf_thres, opt.nms_thres)[0]

        s = '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            with open(str(Path(out)) + opt.txt, 'a') as file:
                file.write(img_name + ' ')
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            for *xyxy, conf, _, cls in det:
                if save_txt:  # Write to file
                    with open('data/clsname2id_all.json') as f:
                        d = json.load(f)
                    cls = classes[int(cls)]
                    cls2id = d[str(cls)]
                    with open(str(Path(out)) + opt.txt, 'a') as file:
                        file.write(('%g ' * 6) % (*xyxy, conf, float(cls2id)))

                if save_img or stream_img:  # Add bbox to image
                    label = '%s %.2f' % (cls, conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls2id)])

            with open(str(Path(out)) + opt.txt, 'a') as file:
                file.write('\n')

        print('%sDone. (%.3fs)' % (s, time.time() - t))

        # Stream results
        if stream_img:
            cv2.imshow(weights, im0)

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-zsd-vs.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/zsd.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='output/ai03_31531_27/best.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='/dlwsdata3/public/ZSD/ZJLAB_ZSD_2019_semifinal_3/ZJLAB_ZSD_2019_semifinal_testset/', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='out29', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--txt', type=str, default='/unseen.txt', help='output txt name')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
