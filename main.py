import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
import time
import numpy as np
from numpy import asarray
# Frequency range for Fast-Fourier Transform
freq_min =1.2 #minimum freq range
freq_max =3 #max freq range

t0 = time.time()
video = cv2.VideoCapture(0)
# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width =int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)
# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
# FourCC is a 4-byte code used to specify the video codec
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),#here MJPG (motion jpeg)
                         25, size)
while (True):
    ret, frame = video.read()

    if ret == True:

        frame = cv2.resize(frame, size)

        # Write the frame into the
        # file 'filename.avi'
        result.write(frame)

        # Display the frame
        # saved in the file
        cv2.imshow('Frame', frame)

        # Press S on keyboard
        # to stop the process
        t1 = time.time()  # current time
        num_seconds = t1 - t0  # diff
        if num_seconds >5:
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Break the loop
    else:
        break
# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")


# Preprocessing phase
print("Reading + preprocessing video...")
video_frames, frame_ct, fps = preprocessing.read_video('filename.avi')


# Build Laplacian video pyramid
print("Building Laplacian video pyramid...")

lap_video = pyramids.build_video_pyramid(video_frames)
amplified_video_pyramid = []

# dim1= len(lap_video)
# dim2=lap_video[0]
# dim3=lap_video[0][0]
# dim4=lap_video[0][0][0]
# dim5=lap_video[0][0][0][0]
#
# print("lap_video dim", dim1, len(dim2),len(dim3),len(dim4),len(dim5))


for i, video in enumerate(lap_video): #video includes all the frames for each level(layer). shape (1,135,500,500,3)[if lev=3]
    # if i == 0 or i==len(lap_video)-1:
    #     continue
    if i != (len(lap_video)-1)//2:
        continue

    # Eulerian magnification with temporal FFT filtering
    print("Running FFT and Eulerian magnification...")
    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    lap_video[i] += result

    # Calculate heart rate
    print("Calculating heart rate...")
    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)

# Collapse laplacian pyramid to generate final video
print("Rebuilding final video...")
# amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

# Output heart rate and final video
print("Heart rate: ", heart_rate, "bpm")
print("Displaying final video...")
# for frames in amplified_frames:
#     cv2.imshow("frames", frames)
#     cv2.waitKey(20)
#


