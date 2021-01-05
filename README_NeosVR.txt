-Navigate to the following url: https://github.com/Ruzert/NeosWCFaceTracking/releases
-Open the last version starting with 'NeosWCFaceTracking' and download it.
-Extract it somewhere on your PC.
-Navigate to the Binary folder.
-Run run.bat and enter the configuration. You may need to select the lowest resolution.
  -Alternatively, run facetrackerNeos.exe if you want the default webcam, the recommended tracking model and a resolution of 640x360 to be used.
-Proceed with the in-game setup: https://imgur.com/a/RUiewxc



--FOR DEVELOPERS:

-Install python 3.6+ with pip: https://www.python.org/downloads/release/python-3610/
-Clone or download the project.
-Install the requirements by opening a console window, navigating to the folder, and executing the following command: 
 pip install -r requirements.txt


-Run facetracker.py, it should show your face and the main landmarks on it. If it works, close it, if it doesn't, check your installation or that your webcam is the default device.
-Run facetrackerNeos.py
-Proceed with the ingame setup: https://imgur.com/a/RUiewxc


#KNOWN ISSUES

- If you've just started tracking, the tracking data might be prone to errors. Try to wait a little for the AI to warm up.

- You may be blinded by your avatar's head when you lean forward, if you wish to reduce this effect you may want to change the near clip of that avatar.
  This video by ProbablePrime explains how to set it up:
  Neos VR Tutorial: Setting up a Near Clip for your Avatars: https://www.youtube.com/watch?v=U1HuZ10M0AQ