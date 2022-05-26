import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import PIL
from PIL import Image as im
import matplotlib.pyplot as plt
import skimage.color
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")#see properly


# def rgb2gray(rgb):
#     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#
#     return gray

# Read in and simultaneously preprocess video
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()

    while cap.isOpened():

        ret, img = cap.read()#ret is boolean, it is true when video stream contains a frame to read. img is a frame of video stream
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#cv2.COLOR_RGB2GRAY-no clue
        yiq=skimage.color.rgb2yiq(img)
        # cv2.imshow("gray", gray)
        # cv2.imshow("yiq",yiq)

        #print("gray", np.shape(gray)) #shape=(480,640)
        #--------------------------------------------------------------------------
        # img[:, :, 2] = 0
        # img[:, :, 0] = 0

        #cv2.imshow('green_img', img)
        #cv2.waitKey()
        roi_frame=img
        #-------------------------------------------------------------------------
        #cv2.imshow("roi_frame", roi_frame)
        #shape=(480,640,3)
        # cv2.imshow("roi_frame", roi_frame)
        #print("roi_frame after gray", np.shape(roi_frame)) #shape=(480,640,3)
        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)# returns position of detected face, if found
                                                                   # detectMultiScale(img,scale_factor, min_neigh

        # Select ROI
        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                roi_frame = img[y:y + h, x:x + w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500,500))
                # print("roi_frame", np.shape(roi_frame)) #roi_frame (500, 500, 3)
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                # print("2 steps before", np.shape(frame)) #roi_frame (500, 500, 3)
                frame[:] = roi_frame * (1. / 255)  #Normalisation
                # print("1 step before video_frame", np.shape(frame)) #frame (500,500,3)
                video_frames.append(frame)


            # gray_image = rgb2gray(frame)
            # cv2.imshow("gray image", gray_image)
            # cv2.imshow("input image", frame)
            # f = np.fft.fft2(gray_image)
            # fshift = np.fft.fftshift(f)
            # magnitude_spectrum = 20 * np.log(np.abs(fshift))
            #
            # plt.subplot(121), plt.imshow(gray_image, cmap='gray')
            # plt.title('Input gray Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
            # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            # plt.show()
    # print('video_frames_shape', np.shape(video_frames)) #video_frames shape = (no_frames,500,500,3)
    frame_ct = len(video_frames)
    cap.release()
    return video_frames, frame_ct, fps
