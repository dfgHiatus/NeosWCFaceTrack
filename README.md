## Overview

NeosWCFaceTrack is a Fork of OpenSeeFace designed to give Screen-Mode users in NeosVR control of their avatar's expressions.  

If you'd like to suggest an improvement or report a bug, feel free to do so in the Issues tab.

## Installation

Please refer to the [Wiki](https://github.com/Ruz-eh/NeosWCFaceTrack/wiki/Installation)!

# For Developers, read on

## Notes and tidbits

There is an inverse relationship between sample speed and sample accuracy. OpenSeeFace leverages several different models for face detection, some are optimal for slower PCs. The models are as follows:

* Model **-1**: The fastest model, though Blink/Gaze tracking are disabled.
* Model **0**
* Model **1**
* Model **2**
* Model **3** : This is the slowest and highest accuracy model.
* Model **4** : Like 3, but optimized for wink detection.

Some Special models that are out of order:
* Model **-3** : Quality is between -1 and 0.
* Model **-2** (default) : Quality is between 1 and 2. Recommended for the most part.

To use these in Python 3.6+, open a terminal windows and type "python facetrackerNeos.py --model X" as an argument, with X being the model number in question.

## Credits

Big thanks to OpenSeeFace for making this possible! 
