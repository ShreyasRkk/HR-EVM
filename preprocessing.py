import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import PIL
from PIL import Image as im
import matplotlib.pyplot as plt
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# Read in and simultaneously preprocess video
def read_video(path):
    c=0
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roi_frame = img

        # Detect face
        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)


        # Select ROI
        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                roi_frame = img[y:y + h, x:x + w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                frame[:] = roi_frame * (1. / 255)
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

    frame_ct = len(video_frames)
    cap.release()
    return video_frames, frame_ct, fps
