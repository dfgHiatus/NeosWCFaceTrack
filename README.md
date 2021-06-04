## Overview

NeosWCFaceTrack is a Fork of OpenSeeFace designed to give Screen-Mode users in NeosVR control of their avatar's expressions.  

If you'd like to suggest an improvement or report a bug, feel free to do so in the Issues tab.

## Installation

Please refer to the [Wiki](https://github.com/Ruz-eh/NeosWCFaceTrack/wiki)!

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

## Running the python script

1. Install python 3.6+ with pip: https://www.python.org/downloads/release/python-3610/ (also works on 3.8)
2. Clone or download the project.
3. Install the requirements by opening a console window, navigating to the folder of the project, and executing the following command: 
 `pip install -r requirements.txt`
4. Make sure the default camera is connected and not in use.
5. To check if everything's working, run facetracker.py, it should show your face and the main landmarks on it. If it works, close it, if it doesn't, check your installation or that your webcam is the default device.
6. Run facetrackerNeos.py while using an avatar that's correctly set up.

## Building the Windows executable

1. Install pyinstaller by running the following command: `pip install pyinstaller`
2. Run the `make_exe.bat` batch script, it should build the binary's folder inside the `dist` folder.
3. You can either run that executable separately or move that folder to the root of the project, as is in the releases. If you do move it, you may delete the `models` folder inside the binary, as it will use the `models` folder located in the root of the project.

## Usage of webcam data

The program collects a single frame from the camera per websocket request, processes it through the AI to determine facial landmarks, and then destroys the frame. Afterwards, the processed landmark data gets sent over to the client of the websockets server in a string format, for the purpose of driving a virtual avatar. No other data is collected, and no other network connections are made.

## Credits

Big thanks to OpenSeeFace for making this possible! 

## License

The code and models are distributed under the BSD 2-clause license.

You can find licenses of third party libraries used for binary builds in the Licenses folder.
