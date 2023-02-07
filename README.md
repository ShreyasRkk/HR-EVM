# Remote Heart Rate Detection through Eulerian Magnification of Face Videos

An implementation of Eulerian video magnification using computer vision. This program uses the method for the application of remotely detecting an individual's heart rate in beats per minute from a still video of his/her face.

Built with OpenCV, NumPy, and SciPy in Python 3

## Program organization:
The main.py file contains the main program that utilizes all of the other modules defined in the other code files
to read in the input video, run Eulerian magnification on it, and to display the results. The purposes of the other
files are described below:
- preprocessing.py - Contains function to read in video from file and uses Haar cascade face detection to select an ROI on all frames
- pyramids.py - Contains functions to generate and collapse image/video pyramids (Gaussian/Laplacian)
- eulerian.py - Contains function for a temporal bandpass filter that uses a Fast-Fourier Transform
- heartrate.py - Contains function to calculate heart rate from FFT results

 To alter the frequency range to be filtered, change the values assigned to freq_min and freq_max on lines 8
and 9.

## How to run 
Type python main.py to run the code.
