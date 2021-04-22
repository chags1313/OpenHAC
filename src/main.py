# Use this for fbs freeze
# from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import cv2
import numpy as np
import time
import sys
from imutils import face_utils
from face_utilities import Face_utilities
from signal_processing import Signal_processing

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pandas as pd
import subprocess 

import scipy
import progressbar
import os
from datetime import datetime
from PIL import Image

from bamlt import Ui_MainWindow
from train_dlg1 import Ui_train_dlg
from deploy_dlg1 import Ui_deploy_dlg
from new_dlg import Ui_new_dlg
from dataframe_dlg import Ui_Dialog
import webbrowser




import pandas as pd
import os
import argparse
import logging
import glob
import time
from os.path import splitext

def hhmmss(ms):
    # s = 1000
    # m = 60000
    # h = 360000
    h, r = divmod(ms, 36000)
    m, r = divmod(r, 60000)
    s, _ = divmod(r, 1000)
    return ("%d:%02d:%02d" % (h,m,s)) if h else ("%d:%02d" % (m,s))

class ViewerWindow(QMainWindow):
        state = pyqtSignal(bool)

        def closeEvent(self, e):
            # Emit the window state, to update the viewer toggle button.
            self.state.emit(False)

        def toggle_viewer(self, state):
            if state:
                self.viewer.show()
            else:
                self.viewer.hide()


class PlaylistModel(QAbstractListModel):
    def __init__(self, playlist, *args, **kwargs):
        super(PlaylistModel, self).__init__(*args, **kwargs)
        self.playlist = playlist

    def data(self, index, role):
        if role == Qt.DisplayRole:
            media = self.playlist.media(index.row())
            return media.canonicalUrl().fileName()

    def rowCount(self, index):
        return self.playlist.mediaCount()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("icons8-baby-face-64.png"))
        

        self.player = QMediaPlayer()
        self.player.error.connect(self.erroralert)

        #self.player.error.connect(self.erroralert)
        self.player.play()

        # Setup the playlist.
        self.playlist = QMediaPlaylist()
        self.player.setPlaylist(self.playlist)

        # Add viewer for video playback, separate floating window.
        self.viewer = ViewerWindow(self)
        self.viewer.setWindowFlags(self.viewer.windowFlags() | Qt.WindowStaysOnTopHint)
        self.viewer.setMinimumSize(QSize(480,360))

        videoWidget = QVideoWidget()
        self.viewer.setCentralWidget(videoWidget)
        self.player.setVideoOutput(videoWidget)

        # Connect control buttons/slides for media player.
        self.playButton.pressed.connect(self.player.play)
        self.pauseButton.pressed.connect(self.player.pause)
        self.stopButton.pressed.connect(self.player.stop)
        self.volumeSlider.valueChanged.connect(self.player.setVolume)

        self.viewButton.toggled.connect(self.toggle_viewer)
        self.viewer.state.connect(self.viewButton.setChecked)

        self.previousButton.pressed.connect(self.playlist.previous)
        self.nextButton.pressed.connect(self.playlist.next)

        self.model = PlaylistModel(self.playlist)
        self.playlistView.setModel(self.model)
        self.playlist.currentIndexChanged.connect(self.playlist_position_changed)
        selection_model = self.playlistView.selectionModel()
        selection_model.selectionChanged.connect(self.playlist_selection_changed)

        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        self.timeSlider.valueChanged.connect(self.player.setPosition)

        self.actionAdd_video.triggered.connect(self.open_file)
        self.actionHeartrate.triggered.connect(self.heart_click)
        self.actionFace_and_head.triggered.connect(self.open_face)
        self.actionPose_analysis.triggered.connect(self.body_click)
        self.actionTrain.triggered.connect(self.trn_dlg)
        #self.actionLoad_class.triggered.connect(self.load_class)
        self.actionDeploy.triggered.connect(self.deploy_dlg)
        self.actionNew_Project.triggered.connect(self.new_dlg)
        self.actionSave_Project.triggered.connect(self.load_prjct)
        #self.actionLoad_Project.triggered.connect(self.load_prjct)
        self.actionDataframeview.triggered.connect(self.df_dlg)
        self.actionAUC.triggered.connect(self.auc_plt)
        self.actionPrecision_Recall_Curve.triggered.connect(self.pr_plt)
        #self.actionConfusion_Matrix_2.triggered.connect(self.cm_plt)
        self.actionFeature_Importance.triggered.connect(self.fi_plt)
        self.actionSummary.triggered.connect(self.sum_plt)
        self.actionCorrelation.triggered.connect(self.cor_plt)
        self.actionReason.triggered.connect(self.r_plt)
        self.actionHelp.triggered.connect(self.help)
        self.actionAbout.triggered.connect(self.about)
        #self.actionHead_motion.triggered.connect(self.head)
        self.setAcceptDrops(True)
  

        self.show()
        
    def about(self):
        webbrowser.open('https://chags1313.github.io/D-BAMLT/')
    def help(self):
        webbrowser.open('https://github.com/chags1313/D-BAMLT/wiki')

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            self.playlist.addMedia(
                QMediaContent(url)
            )

        self.model.layoutChanged.emit()

        # If not playing, seeking to first of newly added + play.
        if self.player.state() != QMediaPlayer.PlayingState:
            i = self.playlist.mediaCount() - len(e.mimeData().urls())
            self.playlist.setCurrentIndex(i)
            self.player.play()

    def open_file(self):
        global path
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "mp3 Audio (*.mp3);mp4 Video (*.mp4);Movie files (*.mov);All files (*.*)")

        if path:
            self.playlist.addMedia(
                QMediaContent(
                    QUrl.fromLocalFile(path)
                )
            )

        self.model.layoutChanged.emit()
        print(path)
 

    def update_duration(self, duration):
        print("!", duration)
        print("?", self.player.duration())
        
        self.timeSlider.setMaximum(duration)

        if duration >= 0:
            self.totalTimeLabel.setText(hhmmss(duration))

    def update_position(self, position):
        if position >= 0:
            self.currentTimeLabel.setText(hhmmss(position))

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.timeSlider.blockSignals(True)
        self.timeSlider.setValue(position)
        self.timeSlider.blockSignals(False)

    def playlist_selection_changed(self, ix):
        # We receive a QItemSelection from selectionChanged.
        i = ix.indexes()[0].row()
        self.playlist.setCurrentIndex(i)

    def playlist_position_changed(self, i):
        if i > -1:
            ix = self.model.index(i)
            self.playlistView.setCurrentIndex(ix)

    def toggle_viewer(self, state):
        if state:
            self.viewer.show()
        else:
            self.viewer.hide()
    '''def save_prjct(self):
        
        
        globals()
        save_prjct_file, _ = QFileDialog.getSaveFileName(self, "Save project file", "", "pkl (*.pkl)")
        dill.dump_session(save_prjct_file)'''
        
    def load_prjct(self):
        self.load_prjct_file, _ = QFileDialog.getOpenFileName(self, "Open existing project file", "", "txt (*.txt)")
        print(self.load_prjct_file)


       
            


    def erroralert(self, *args):
        print(args)

    def deploy_dlg(self):
        global loaded_data
        deploy=deploy_dlg(self)
        deploy.exec()

        
    def new_dlg(self):
        global loaded_data
        new=new_dlg(self)
        new.exec()

        

    def heart_click(self):
        hrt=heartrate(self)
        #hrt.exec()
        #self.hrt_used = print('Digital biomarkers for heartrate analyzed.' + " " + date_time)

    def body_click(self):
        bp = body_pose(self)
        #hrt.exec()
        #self.body_click_used = print("Digitial biomarkers for body keypoints analyzed." + " " + date_time)

    def open_face(self):
        work_d = os.getcwd()
        #work_d = work_d + "/face"
        # Use this for OpenFace variables
        # Clone OpenFace - follow documentation on OpenFace to download models 
        ####subprocess.run(["open_dbm/OpenFace_2.2.0_win_x64/FaceLandmarkVidMulti.exe","-f", path, "-vis-aus", "-vis-track"])
        # Use this for OpenDBM variables - must have OpenFace directory in process_data file and must have models downloaded
        ####subprocess.run(["python", "open_dbm/process_data1.py","--input_path", path, "--output_path", work_d])
        print("Use this function for face head and eye biomarkers")
    def head(self):
        work_d = os.getcwd()
        #work_d = work_d + "/head_mvt"
        #movement = "'{}".format('movement')
        ### see open_face above for how to implement 
        ###subprocess.run(["python", "open_dbm/process_data1.py","--input_path", path, "--output_path", work_d])
        print("Use this function for head motion and occulomotor biomarkers")
    def trn_dlg(self):
        trn_dlg = train_dlg(self)
        trn_dlg.exec() 
    def click_data(self):
        self.data = QFileDialog.getOpenFileName(self, "Open file", "", "csv (*.csv)")
        self.data = pd.read_csv(self.data)
    def load_class(self):
        global loaded_data
        loaded_data, _ = QFileDialog.getOpenFileName(self, "Open file for deployment predictions", "", ".csv (*.csv)")
    def df_dlg(self):
        dfdlg = df_dlg(self)
        dfdlg.exec() 
    def auc_plt(self):
        ## Plots only open after training is complete
        image = Image.open('AUC.png')
        image.show()
    def pr_plt(self):

        image = Image.open('Precision Recall.png')
        image.show()
    def cm_plt(self):

        image = Image.open('Confusion Matrix.png')
        image.show()
    def lc_plt(self):

        image = Image.open('Learning Curve.png')
        image.show()
    def fi_plt(self):

        image = Image.open('Feature Importance.png')
        image.show()
    def rfs_plt(self):
  
        image = Image.open('Recursive Feature Selection.png')
        image.show()
    def cpe_plt(self):

        image = Image.open('Prediction Error.png')
        image.show()
    def sum_plt(self):

        image = Image.open('SHAP summary.png')
        image.show()
    def cor_plt(self):

        image = Image.open('SHAP correlation.png')
        image.show()
    def r_plt(self):

        image = Image.open('SHAP reason.html')
        image.show()    



class heartrate():
   def __init__(self, heartrate, *args, **kwargs):

    cap = cv2.VideoCapture(path)
    
    fu = Face_utilities()
    sp = Signal_processing()
    
    i=0
    last_rects = None
    last_shape = None
    last_age = None
    last_gender = None
    
    face_detect_on = False
    age_gender_on = False

    t = time.time()
    
    #for signal_processing
    BUFFER_SIZE = 100
    
    fps=0 #for real time capture
    video_fps = cap.get(cv2.CAP_PROP_FPS) # for video capture
    print(video_fps)
    
    times = []
    data_buffer = []
    
    # data for plotting
    filtered_data = []
    
    fft_of_interest = []
    freqs_of_interest = []
    
    bpm = 0
    
    #plotting
    app = QtGui.QApplication([])  
    
    win = pg.GraphicsWindow(title="plotting")
    p1 = win.addPlot(title="FFT")
    p2 = win.addPlot(title ="Signal")
    win.resize(1200,600)
    
    def update():
        p1.clear()
        p1.plot(np.column_stack((freqs_of_interest,fft_of_interest)), pen = 'g')
        
        p2.clear()
        p2.plot(filtered_data[20:],pen='g')
        app.processEvents()
    
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(300)
    
    while True:
        # grab a frame -> face detection -> crop the face -> 68 facial landmarks -> get mask from those landmarks

        # calculate time for each loop
        t0 = time.time()
        
        if(i%1==0):
            face_detect_on = True
            if(i%10==0):
                age_gender_on = True
            else:
                age_gender_on = False
        else: 
            face_detect_on = False
        
        ret, frame = cap.read()
        #frame_copy = frame.copy()
        
        if frame is None:
            print("End of video")
            cv2.destroyAllWindows()
            timer.stop()
            #sys.exit()
            break
        
        #display_frame, aligned_face = flow_process(frame)
        
        
        ret_process = fu.no_age_gender_face_process(frame, "68")
        
        if ret_process is None:
            cv2.putText(frame, "No face detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            cv2.imshow("frame",frame)
            print(time.time()-t0)
            
            cv2.destroyWindow("face")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                timer.stop()
                #sys.exit()
                break
            continue
        
        rects, face, shape, aligned_face, aligned_shape = ret_process
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        #overlay_text = "%s, %s" % (gender, age)
        #cv2.putText(frame, overlay_text ,(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        
        if(len(aligned_shape)==68):
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
        else:
            #print(shape[4][1])
            #print(shape[2][1])
            #print(int((shape[4][1] - shape[2][1])))
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
        
        for (x, y) in aligned_shape: 
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)
            
            
        #for signal_processing
        ROIs = fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = sp.extract_color(ROIs)
        print(green_val)
        
        data_buffer.append(green_val)
        
        times.append(time.time() - t)

        
        L = len(data_buffer)
        #print("buffer length: " + str(L))
        
        if L > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]
            times = times[-BUFFER_SIZE:]
            #bpms = bpms[-BUFFER_SIZE//2:]
            L = BUFFER_SIZE
        #print(times)
        if L==100:
            fps = float(L) / (times[-1] - times[0])
            cv2.putText(frame, "fps: {0:.2f}".format(fps), (30,int(frame.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            #
            detrended_data = sp.signal_detrending(data_buffer)
            #print(len(detrended_data))
            #print(len(times))
            interpolated_data = sp.interpolation(detrended_data, times)
            
            normalized_data = sp.normalization(interpolated_data)
            
            fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)
            
            max_arg = np.argmax(fft_of_interest)
            bpm = freqs_of_interest[max_arg]
            cv2.putText(frame, "HR: {0:.2f}".format(bpm), (int(frame.shape[1]*0.8),int(frame.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            #print(detrended_data)
            filtered_data = sp.butter_bandpass_filter(interpolated_data, (bpm-20)/60, (bpm+20)/60, fps, order = 3)
            #print(fps)
            #filtered_data = sp.butter_bandpass_filter(interpolated_data, 0.8, 3, fps, order = 3)
            
        #write to txt file
        with open("dbalt_hr_output.txt",mode = "a+") as f:
            f.write("time: {0:.4f} ".format(times[-1]) + ", HR: {0:.2f} ".format(bpm) + "\n")
                        
        # display
        cv2.imshow("frame",frame)
        cv2.imshow("face",aligned_face)
        #cv2.imshow("mask",mask)
        i = i+1
        print("time of the loop number "+ str(i) +" : " + str(time.time()-t0))
        
        # waitKey to show the frame and break loop whenever 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            timer.stop()
            #sys.exit()
            break
        
        
    cap.release()
    cv2.destroyAllWindows()

    print("total running time: " + str(time.time() - t)) 
    
    output_conv = pd.read_csv("dbalt_hr_output.txt")
    output_conv.to_csv('dbalt_hr_output.csv', index=None)




class body_pose():
   def __init__(self, body_pose, *args, **kwargs):
# Necessary Paths
## Download these and specify file paths
    protoFile = "Pose-Estimation-Clean-master/models/pose/mpi/pose_deploy_linevec.prototxt"
    weightsFile = "Pose-Estimation-Clean-master/models/pose/mpi/pose_iter_160000.caffemodel"
    global path
    video_path = path
    csv_path = 'body_pose_output.csv'

    # Load the model and the weights
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Store the input video specifics
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()
    #(frameHeight, frameWidth) = cap.frame.shape[:2]
    #h = 500
    #w = int((h/frameHeight) * frameWidth)
    h = 500
    w = 890

    # Dimensions for inputing into the model
    inHeight = 368
    inWidth = 368

    # Set up the progressbar
    widgets = ["--[INFO]-- Analyzing Video: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = n_frames,
                                   widgets=widgets).start()
    p = 0

    data = []
    previous_x, previous_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # Define the output
    out_path = 'outputs/out_11.mp4'
    output = cv2.VideoWriter(out_path, 0, fps, (w, h))

    fourcc = cv2.VideoWriter_fourcc(*'MP44')
    writer = None
    (f_h, f_w) = (h, w)
    zeros = None

    # There are 15 points in the skeleton
    pairs = [[0,1], # head
             [1,2],[1,5], # sholders
             [2,3],[3,4],[5,6],[6,7], # arms
             [1,14],[14,11],[14,8], # hips
             [8,9],[9,10],[11,12],[12,13]] # legs

    # probability threshold fro prediction of the coordinates
    thresh = 0.4

    circle_color, line_color = (0,255,255), (0,255,0)

    # Start the iteration
    while True:
        ok, frame = cap.read()

        if ok != True:
            break
    
        frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)    
        frame_copy = np.copy(frame)
    
        # Input the frame into the model
        inpBlob = cv2.dnn.blobFromImage(frame_copy, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
    
        H = output.shape[2]
        W = output.shape[3]
    
        points = []
        x_data, y_data = [], []
    
        # Iterate through the returned output and store the data
        for i in range(15):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (w * point[0]) / W
            y = (h * point[1]) / H
        
            if prob > thresh:
                points.append((int(x), int(y)))
                x_data.append(x)
                y_data.append(y)
            else :
                points.append((0, 0))
                x_data.append(previous_x[i])
                y_data.append(previous_y[i])
    
        for i in range(len(points)):
            cv2.circle(frame_copy, (points[i][0], points[i][1]), 2, circle_color, -1)
    
        for pair in pairs:
            partA = pair[0]
            partB = pair[1]
            cv2.line(frame_copy, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps,
                                     (f_w, f_h), True)
            zeros = np.zeros((f_h, f_w), dtype="uint8")
    
        writer.write(cv2.resize(frame_copy,(f_w, f_h)))
    
        cv2.imshow('Body pose analysis' ,frame_copy)
    
        data.append(x_data + y_data)
        previous_x, previous_y = x_data, y_data
    
        p += 1
        pbar.update(p)
    
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
            break

    # Save the output data from the video in CSV format
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index = False)
    print('save complete')

    pbar.finish()
    cap.release()
    cv2.destroyAllWindows()
    
class train_dlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_train_dlg()
        self.ui.setupUi(self)
        
class deploy_dlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_deploy_dlg()
        self.ui.setupUi(self)
        
class new_dlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_new_dlg()
        self.ui.setupUi(self)
        

        
class df_dlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        






if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("D-BAMLT")
    app.setStyle("Fusion")

    # Fusion dark palette from https://gist.github.com/QuantumCD/6245215.
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

    window = MainWindow()
    app.exec_()


