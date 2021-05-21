# Overview

NeosWCFaceTrack is a Fork of OpenSeeFace designed to give Screen-Mode users in NeosVR control of their avatar's expressions. 

# Installation

Please refer to the Wiki! We have also thoroughly documented the process as well on [Imgur](https://imgur.com/a/RUiewxc), be sure to take a look there as well.

# Notes and tidbits

OpenSeeFace leverages several different models for face detection, making it optimal for slower PCs.
Available models are:

* Model **-1**: The fastest model, with the poorest quality. Blink and gaze tracking are disabled.
* Model **0**: This is a very fast, low accuracy model.
* Model **1**: This is a slightly slower model with better accuracy.
* Model **2**: This is a slower model with good accuracy.
* Model **3** : This is the slowest and highest accuracy model.
* Model **4** : Like 3, but optimized for wink detection.
* Model **-3** : Quality is between -1 and 0.
* Model **-2** (default) : Quality is roughly like 1, but is faster. Recommended unless you want better tracking accuracy, in which case use model 2, 3, or 4.

You can select a model by running run.bat located inside the Binary folder of a release.

To use these in Python, open a terminal windows and type "python facetrackerNeos.py --model X" as an argument, with X being the model in question.

# Credits

Big thanks to OpenSeeFace for making this possible! 
