# Overview

NeosWCFaceTrack is a Fork of OpenSeeFace designed to give Screen-Mode users in NeosVR control of their avatar's expressions. 

# Installation

Please refer to the Wiki! We have also thoroughly documented the process as well on [Imgur](https://imgur.com/a/RUiewxc), be sure to take a look there as well.

# Notes and tidbits

OpenSeeFace leverages several different models for face detection, making it optimal for slower PCs. We have omitted Model -1 as it lacks gaze tracking. Straight from OpenSeeFace's README:

* Model **0**: This is a very fast, low accuracy model. (68fps)
* Model **1**: This is a slightly slower model with better accuracy. (59fps)
* Model **2**: This is a slower model with good accuracy. (50fps)
* Model **3** (default): This is the slowest and highest accuracy model. (44fps)

To use these, open a terminal windows and type "python faceTrackerNeos.py --model X" as an argument, with X being the model in question. 

# Credits

Big thanks to OpenSeeFace for making this possible! 
