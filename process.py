import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from util.datasets import *
from util.utils import *
from models.LPRNet import *

import os
import sys

import cv2 as cv
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import*
from PyQt5.QtWidgets import*
from PyQt5.QtCore import*
from threading import  Thread
from PyQt5.QtCore import pyqtSignal,QObject


import main_ui

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MyThread(QThread):
    my_str = pyqtSignal(str) # 创建任务信号

    def run(self):
        self.my_str.emit("ok") # 发出任务完成信号


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.ui = main_ui.Ui_MainWindow()
        self.ui.setupUi(self)

        self.my_thread = MyThread()



        self.folder_dir = "" #当前文件夹
        self.file_paths = []  # 文件列表
        self.file_index = 0  # 文件索引
        self.num_trage = 0
        self.num = 0
        self.end_click = 0
        self.recfile_path = []
        self.setstyle()
        self.bind()


    def get_sin_out(self, out_str):
        """
        :param out_str:
        :return:
        """
        if out_str == "ok":
            print("处理完成")


    def setstyle(self):
        self.ui.verticalWidget.setStyleSheet("border:3px solid red")
        self.ui.verticalWidget_2.setStyleSheet("border:3px solid red")
        self.ui.verticalWidget_3.setStyleSheet("border:3px solid red")

        img1 = QtGui.QPixmap('BG.jpg').scaled(361, 480)
        img2 = QtGui.QPixmap('BG_2.jpg').scaled(631, 201)
        self.ui.label_6.setPixmap(img1)
        self.ui.label_9.setPixmap(img1)
        self.ui.label_19.setPixmap(img2)



    def bind(self):
        self.ui.pushButton.clicked.connect(self.on_btnFolderPrevious_clicked)
        self.ui.pushButton_3.clicked.connect(self.on_btnImportFolder_clicked)
        self.ui.pushButton_2.clicked.connect(self.on_btnFolderNext_clicked)
        self.ui.pushButton_4.clicked.connect(self.end_thread)
        self.ui.pushButton_5.clicked.connect(self.start_thread)
        self.my_thread.my_str.connect(self.get_sin_out)
    def end_thread(self):
        self.end_click = 1

    def start_thread(self):
        """
        启动多线程
        :return:
        """
        try:
            while (self.file_index < len(self.file_paths)):

                self.file_index += 1

                if len(self.file_paths) <= 0 or self.file_index >= len(self.file_paths) or self.end_click == 1:
                    self.end_click = 0
                    self.file_index -= 1
                    self.my_thread.start()
                    return

                cur_path = self.file_paths[self.file_index]
                img = QtGui.QPixmap(cur_path).scaled(self.ui.label_6.width(), self.ui.label_6.height())
                self.ui.label_6.setPixmap(img)
                setparser(self, cur_path)
                self.recfile_path.clear()
                for root, dirs, files in os.walk(
                        'D:/Temp_Pic/',
                        topdown=False):
                    for file in files:
                        self.recfile_path.append(root + file)

                rec_img = QtGui.QPixmap(self.recfile_path[0]).scaled(self.ui.label_9.width(), self.ui.label_9.height())

                self.ui.label_9.setPixmap(rec_img)


                ACC = float(self.num_trage / self.num)

                self.ui.label_17.setText(str(ACC))
                QApplication.processEvents()
                # time.sleep(0.5)


            self.my_thread.start()
        except Exception as e:
            print(e)


    def showresult(self,text,conf):
        self.ui.label_13.setText(text)
        self.ui.label_15.setText(conf)
        self.ui.label_16.setText(str(self.file_index + 1) + ' / ' + str(len(self.file_paths)))





    def on_btnImportFolder_clicked(self):

        cur_dir = QDir.currentPath()  # 获取当前文件夹路径
        # 选择文件夹
        dir_path = QFileDialog.getExistingDirectory(self, '打开文件夹', cur_dir)
        self.folder_dir = dir_path

        # 读取文件夹文件
        self.file_paths.clear()
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file in files:
                self.file_paths.append(root + '/' + file)



        if len(self.file_paths) <= 0:
            return

        # # 获取第一个文件
        self.file_index = 0

        self.num_trage = 0
        self.num = 0
        cur_path = self.file_paths[self.file_index]



        # 处理文件
        img = QtGui.QPixmap(cur_path).scaled(self.ui.label_6.width(), self.ui.label_6.height())
        self.ui.label_6.setPixmap(img)
        setparser(self,cur_path)

        self.recfile_path.clear()


        for root, dirs, files in os.walk(
                'D:/Temp_Pic/',
                topdown=False):
            for file in files:
                self.recfile_path.append(root + file)


        rec_img = QtGui.QPixmap(self.recfile_path[0]).scaled(self.ui.label_9.width(), self.ui.label_9.height())

        self.ui.label_9.setPixmap(rec_img)

   

    def on_btnFolderNext_clicked(self):
        # 文件索引累加 1
        self.file_index += 1
        if self.file_index >= len(self.file_paths):
            self.file_index = len(self.file_paths) - 1

        if len(self.file_paths) <= 0 or self.file_index >= len(self.file_paths):
            return

        cur_path = self.file_paths[self.file_index]
        img = QtGui.QPixmap(cur_path).scaled(self.ui.label_6.width(), self.ui.label_6.height())
        self.ui.label_6.setPixmap(img)
        setparser(self,cur_path)

        self.recfile_path.clear()
        for root, dirs, files in os.walk(
                'D:/Temp_Pic/',
                topdown=False):
            for file in files:
                self.recfile_path.append(root + file)

        rec_img = QtGui.QPixmap(self.recfile_path[0]).scaled(self.ui.label_9.width(), self.ui.label_9.height())

        self.ui.label_9.setPixmap(rec_img)




    def on_btnFolderPrevious_clicked(self):
        # 文件索引减 1
        self.file_index -= 1
        if self.file_index < 0:
            self.file_index = 0

        if len(self.file_paths) <= 0 or self.file_index >= len(self.file_paths):
            return

        # 当前路径
        cur_path = self.file_paths[self.file_index]
        img = QtGui.QPixmap(cur_path).scaled(self.ui.label_6.width(), self.ui.label_6.height())
        self.ui.label_6.setPixmap(img)
        setparser(self,cur_path)

        self.recfile_path.clear()
        for root, dirs, files in os.walk(
                'D:/Temp_Pic/',
                topdown=False):
            for file in files:
                self.recfile_path.append(root + file)

        rec_img = QtGui.QPixmap(self.recfile_path[0]).scaled(self.ui.label_9.width(), self.ui.label_9.height())

        self.ui.label_9.setPixmap(rec_img)
    def cheek_correct(self,file,text):


        provincelist = [
            "皖", "沪", "津", "渝", "冀",
            "晋", "蒙", "辽", "吉", "黑",
            "苏", "浙", "京", "闽", "赣",
            "鲁", "豫", "鄂", "湘", "粤",
            "桂", "琼", "川", "贵", "云",
            "西", "陕", "甘", "青", "宁",
            "新"]

        wordlist = [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "J", "K",
            "L", "M", "N", "P", "Q",
            "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "0",
            "1", "2", "3", "4", "5",
            "6", "7", "8", "9"]
        imgname = os.path.basename(file).split('.')[0]
        _, _, box, points, label, brightness, blurriness = imgname.split('-')

        # --- 边界框信息
        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]

        # --- 关键点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        # points = points[-2:] + points[:2]

        # --- 读取车牌号
        label = label.split('_')
        # 省份缩写
        province = provincelist[int(label[0])]
        # 车牌信息
        words = [wordlist[int(i)] for i in label[1:]]
        # 车牌号
        label = province + ''.join(words)

        self.ui.label_12.setText(label)
        if text == label:
            self.ui.label_14.setText("正确")
            if self.num <= self.file_index:
                self.num_trage  += 1
        else:
            self.ui.label_14.setText("错误")
        if self.num <= self.file_index:
            self.num += 1
        assert os.path.exists(file), "image file {} dose not exist.".format(file)
        xmin = points[2][0]
        xmax = points[0][0]
        ymin = points[2][1]
        ymax = points[0][1]
        img = cv2.imread(file)
        img = Image.fromarray(img)
        img = img.crop((xmin, ymin, xmax, ymax))  # 裁剪出车牌位置
        img = img.resize((940, 240), Image.LANCZOS)
        img = np.asarray(img)  # 转成array,变成24*94*3

        cv2.imencode('.jpg', img)[1].tofile(
            r"D:\Temp_Card\{}.jpg".format(label))
        print(label)
        if(label != ""):
            tmp_img = QtGui.QPixmap( r"D:\Temp_Card\{}.jpg".format(label)).scaled(self.ui.label_19.width(), self.ui.label_19.height())
            self.ui.label_19.setPixmap(tmp_img)






def detect(self, opt):
    classify, out, source, det_weights, rec_weights, view_img, save_txt, imgsz = \
        opt.classify, opt.output, opt.source, opt.det_weights, opt.rec_weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov5 model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier  也就是rec 字符识别
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu')))

        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size demo
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred, plat_num = apply_classifier(pred, modelc, img, im0s)

        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de, lic_plat in zip(det, plat_num):
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls = de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        lb = ""
                        for a, i in enumerate(lic_plat):
                            # if a ==0:
                            #     continue
                            lb += CHARS[int(i)]
                        label = '%s %.2f' % (lb, conf)
                        self.showresult(lb, '%.2f' % conf)
                        cur_path = self.file_paths[self.file_index]
                        self.cheek_correct(cur_path, lb)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (demo + NMS)


            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # rec_result video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:

        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)







def setparser(self,src):
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
    parser.add_argument('--det-weights', nargs='+', type=str, default='./weights/yolov5_best.pt', help='model.pt path(s)')
    parser.add_argument('--rec-weights', nargs='+', type=str, default='./weights/lprnet_best.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=src, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='D:/Temp_Pic/' , help='rec_result folder')  # rec_result folder
    parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented demo')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect(self,opt)
                create_pretrained(opt.weights, opt.weights)
        else:
            detect(self,opt)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()

    sys.exit(app.exec_())
