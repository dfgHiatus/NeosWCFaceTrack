%ECHO OFF

facetrackerNeos -l 1

echo Make sure that nothing is accessing your camera before you proceed.

set cameraNum=0
set /p cameraNum=Select your camera from the list above and enter the corresponding number:
echo.

facetrackerNeos -a %cameraNum%

set dcaps=-1
set /p dcaps=Select your camera mode. It's recommended to pick the lowest resolution:
echo.

echo Tracking model options.
echo Higher numbers generally offer better tracking quality, but slower speed.
echo.
echo -1: Poorest quality tracking but great performance, blinking is disabled.
echo  0: Blinking is enabled in the following models.
echo  1: Better quality than 0, but slower speed.
echo  2: Better quality than 1, but slower speed.
echo  3: Best tracking quality of all the models, but also the slowest model.
echo -3: Quality is between -1 and 0.
echo -2: Quality is roughly like 1, but is faster. Recommended unless you want better tracking accuracy, in which case use model 2 or 3.

set model=-2
set /p model=Select the tracking model (default -2): 
echo.

set smoothr=1
set /p smoothr=Enter 1 to enable smoothing of head rotation, or enter 0 to disable (default 1): 
echo.

set smoothp=1
set /p smoothp=Enter 1 to enable smoothing of head position, or enter 0 to disable (default 1): 
echo.

start facetrackerNeos -c %cameraNum% -D %dcaps% --model %model% --smooth-rotation %smoothr% --smooth-translation %smoothp%