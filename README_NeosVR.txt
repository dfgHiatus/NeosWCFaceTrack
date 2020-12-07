
-Spawn the zip file by double clicking it
-Click it and select the Export option, the file would appear in your Documents/NeosVR folder

-Install python 3.6+ with pip: https://www.python.org/downloads/release/python-3610/
-Clone and install emilianavt's OpenSeeFace github project: https://github.com/emilianavt/OpenSeeFace
-Extract the contents of facetrackingNeos.zip inside that folder
-Install the requirements by opening a cmd window, navigating to the folder, and executing the following command: pip install -r requirements.txt


-Run facetracker.py, it should show your face and the main landmarks on it. If it works, close it, if it doesn't, check your installation or that your webcam is the default device.
-Run facetrackerNeos.py
-Proceed with the ingame setup: https://imgur.com/a/RUiewxc


#KNOWN ISSUES

- The tracking can fail to start tracking an eye blinking, this can happen if you've just started tracking. Try to wait a little for the AI to warm up.
  It can rarely get stuck and never update though, if this happens restart facetrackerNeos.py

- You may be blinded by your avatar's head when you lean forward, if you wish to reduce this effect you may want to change the near clip of that avatar.
  This video by ProbablePrime explains how to set it up:
  Neos VR Tutorial: Setting up a Near Clip for your Avatars: https://www.youtube.com/watch?v=U1HuZ10M0AQ