import copy
import os
import sys
import argparse
import traceback
import gc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=480)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=18)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=480)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=1)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=10)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed. Models 1 and 0 tend to be too rigid for expression and blink detection.", default=2, choices=[-1, 0, 1, 2, 3])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

class OutputLog(object):
    def __init__(self, fh, output):
        self.fh = fh
        self.output = output
    def write(self, buf):
        if not self.fh is None:
            self.fh.write(buf)
        self.output.write(buf)
        self.flush()
    def flush(self):
        if not self.fh is None:
            self.fh.flush()
        self.output.flush()
output_logfile = None
if args.log_output != "":
    output_logfile = open(args.log_output, "w")
sys.stdout = OutputLog(output_logfile, sys.stdout)
sys.stderr = OutputLog(output_logfile, sys.stderr)

if os.name == 'nt':
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    if not args.blackmagic_options is None:
        dshowcapture.set_options(args.blackmagic_options)
    if not args.priority is None:
        import psutil
        classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
        p = psutil.Process(os.getpid())
        p.nice(classes[args.priority])

if os.name == 'nt' and (args.list_cameras > 0 or not args.list_dcaps is None):
    cap = dshowcapture.DShowCapture()
    info = cap.get_info()
    unit = 10000000.;
    if not args.list_dcaps is None:
        formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
        for cam in info:
            if args.list_dcaps == -1:
                type = ""
                if cam['type'] == "Blackmagic":
                    type = "Blackmagic: "
                print(f"{cam['index']}: {type}{cam['name']}")
            if args.list_dcaps != -1 and args.list_dcaps != cam['index']:
                continue
            for caps in cam['caps']:
                format = caps['format']
                if caps['format'] in formats:
                    format = formats[caps['format']]
                if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                else:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
    else:
        if args.list_cameras == 1:
            print("Available cameras:")
        for cam in info:
            type = ""
            if cam['type'] == "Blackmagic":
                type = "Blackmagic: "
            if args.list_cameras == 1:
                print(f"{cam['index']}: {type}{cam['name']}")
            else:
                print(f"{type}{cam['name']}")
    cap.destroy_capture()
    sys.exit(0)

import numpy as np
import time
import cv2
import requests
import asyncio
import websockets
import struct
import json
from OneEuroFilter import LowPassFilter, OneEuroFilter
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")

fps = 0
dcap = None
use_dshowcapture_flag = False
if os.name == 'nt':
    fps = args.fps
    dcap = args.dcap
    use_dshowcapture_flag = True if args.use_dshowcapture == 1 else False
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag)
if type(input_reader.reader) == VideoReader:
    fps = 0

log = None
out = None
first = True
height = 0
width = 0
tracker = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0

features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

if args.log_data != "":
    log = open(args.log_data, "w")
    log.write("Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,Euler.X,Euler.Y,Euler.Z,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
    for i in range(66):
        log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
    for i in range(66):
        log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
    for feature in features:
        log.write(f",{feature}")
    log.write("\r\n")
    log.flush()

is_camera = args.capture == str(try_int(args.capture))

try:
    attempt = 0
    frame_time = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1. / float(fps)
    repeat = args.repeat_video != 0 and type(input_reader.reader) == VideoReader
    need_reinit = 0
    failures = 0
    source_name = input_reader.name

except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")
    input_reader.close()
    if not out is None:
        out.release()
    cv2.destroyAllWindows()



async def facetrack(websocket,path):
    global input_reader
    global need_reinit
    global args
    global fps
    global target_duration
    global use_dshowcapture_flag
    global dcap
    global source_name
    global failures
    global repeat
    global frame_time
    global frame_count
    global attempt
    global is_camera
    global socketString
    global log
    global out
    global first
    global height
    global width
    global tracker
    global features
    global current_features
    global now
    
    oeMouthWideConfig = {
        'freq': 18,
        'mincutoff': 1.0,
        'beta': 0.5,
        'dcutoff': 1.0,
        }
        
    oeEyeBrowUDConfig = {
        'freq': 18,
        'mincutoff': 1.0,
        'beta': 0.5,
        'dcutoff': 1.0,
        }
        
    oeHeadRotationConfig = {
        'freq': 18,
        'mincutoff': 1.0,
        'beta': 0.8,
        'dcutoff': 1.0,
        }
        
    oeHeadPositionConfig = {
        'freq': 18,
        'mincutoff': 0.8,
        'beta': 0.5,
        'dcutoff': 1.0,
        }
    
    # Head Rotation Filters
    # HRotXF = OneEuroFilter(**oeHeadRotationConfig)
    # HRotYF = OneEuroFilter(**oeHeadRotationConfig)
    # HRotZF = OneEuroFilter(**oeHeadRotationConfig)
    # HRotWF = OneEuroFilter(**oeHeadRotationConfig)
    # Head Position FIlters
    HPosXF = OneEuroFilter(**oeHeadPositionConfig)
    HPosYF = OneEuroFilter(**oeHeadPositionConfig)
    HPosZF = OneEuroFilter(**oeHeadPositionConfig)
    
    MouthWideF = OneEuroFilter(**oeMouthWideConfig)
    REyeBrowUDF = OneEuroFilter(**oeEyeBrowUDConfig)
    LEyeBrowUDF = OneEuroFilter(**oeEyeBrowUDConfig)

    lastSocketString = ""
    lastDetected = False
    
    timestamp = 1
    now = time.time()

    async for message in websocket:
        if not input_reader.is_open() or need_reinit == 1:
            input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
            if input_reader.name != source_name:
                print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                sys.exit(1)
            need_reinit = 2
            time.sleep(0.01)
            continue
        if not input_reader.is_ready():
            time.sleep(0.01)
            continue

        # Empty string to be sent
        socketString = ""

        ret, frame = input_reader.read()
        if not ret:
            if repeat:
                if need_reinit == 0:
                    need_reinit = 1
                continue
            elif is_camera:
                attempt += 1
                if attempt > 30:
                    break
                else:
                    time.sleep(0.01)
                    if attempt == 3:
                        need_reinit = 1
                    continue
            else:
                break;

        attempt = 0
        need_reinit = 0
        frame_count += 1
        timestamp += time.time() - now
        now = time.time()

        if first:
            first = False
            height, width, channels = frame.shape
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model >= 0 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)

        try:
            faces = tracker.predict(frame)
            detected = False
            for face_num, f in enumerate(faces):
                f = copy.copy(f)
                f.id += args.face_id_offset
                if f.eye_blink is None:
                    f.eye_blink = [1, 1]
                right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                if args.silent == 0:
                    print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
                detected = True

                # Available properties: now f.id width height f.eye_blink[0] f.eye_blink[1] f.success f.pnp_error f.quaternion[0] f.quaternion[1] f.quaternion[2] f.quaternion[3] f.euler[0] f.euler[1] f.euler[2] f.translation[0] f.translation[1] f.translation[2]

                # Socketstring concatenation for output into neos
                
                # Add the rotation floatQ
                # Use this line and comment the other one if you wanna have jitter-free rotation in exchange for a latency increase and a reduction in responsiveness to fast motions
                #socketString += f"[{HRotXF(f.quaternion[0],timestamp)};{HRotYF(f.quaternion[1],timestamp)};{HRotZF(f.quaternion[2],timestamp)};{HRotWF(f.quaternion[3],timestamp)}],"
                socketString += f"[{f.quaternion[0]};{f.quaternion[1]};{f.quaternion[2]};{f.quaternion[3]}],"
                
                # Add the translation float3
                try:
                    socketString += f"[{HPosXF(f.translation[1], timestamp):.6f};{HPosYF(f.translation[0], timestamp):.6f};{HPosZF(f.translation[2], timestamp):.6f}],"
                except:
                    socketString += f"[{f.translation[1]:.6f};{f.translation[0]:.6f};{f.translation[2]:.6f}],"
                  
                # Add the blink ratio, output the most opened eye's ratio for both if the difference between
                #eye ratios is less than the threshold, otherwise pass the most opened eye's ratio and
                #multiply the blinking eye's ratio to make winking possible
                if f.eye_blink[1]-f.eye_blink[0] > 0.17:
                    socketString += f"[{f.eye_blink[1]:.6f};{0.5*f.eye_blink[0]*f.eye_blink[0]:.6f}],"
                elif f.eye_blink[1]-f.eye_blink[0] < -0.17:
                    socketString += f"[{0.5*f.eye_blink[1]*f.eye_blink[1]:.6f};{f.eye_blink[0]:.6f}],"
                else:
                    avgBlink = (f.eye_blink[0] + f.eye_blink[1] + max(f.eye_blink[0],f.eye_blink[1])) / 3
                    socketString += f"[{avgBlink:.6f};{avgBlink:.6f}],"
                    
                # Features: ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

                # Features can return a null value, we don't want this so we check if they are null and set them to 0.0
                if (not "mouth_open" in f.current_features) or (f.current_features["mouth_open"] < 0.15):
                    mouthOpen = 0.0
                else:
                    mouthOpen = (f.current_features["mouth_open"] - 0.15) * 1.1764
                if (not "mouth_wide" in f.current_features):
                    f.current_features["mouth_wide"] = 0.0
                # Add the mouth properties as float2, a positive number means more open
                socketString += f"[{mouthOpen:.6f};{MouthWideF(f.current_features['mouth_wide'], timestamp):.6f}],"

                if not "eyebrow_updown_l" in f.current_features:
                    f.current_features["eyebrow_updown_l"] = 0.0
                if not "eyebrow_updown_r" in f.current_features:
                    f.current_features["eyebrow_updown_r"] = 0.0
                # Add both eyebrow's up/down ratio as float2, a positive number means up
                socketString += f"[{LEyeBrowUDF(f.current_features['eyebrow_updown_l'], timestamp):.6f};{REyeBrowUDF(f.current_features['eyebrow_updown_r'], timestamp):.6f}],"

                # If the AI loses tracking of a face, when it recovers it it'll return 0.0 on all data for the first frame
                # This fixes it by sending the last good tracking data instead
                if (f.euler[0] is None) or f.euler[0] == 0.0:
                    socketString = lastSocketString

            if detected:
                if not lastDetected:
                    #Prints on console every time the AI loses/recovers tracking of a face
                    print("Got Tracking")
                    lastDetected = True
                # This is valid data, so we store it as the last valid data
                lastSocketString = socketString
                # The boolean value is meant for Logix, usually returning the data to a neutral state
                await websocket.send(f"True,{socketString}")
            else:
                if lastDetected:
                    print("Lost Tracking")
                    lastDetected = False
                # We send the last valid data as to not make the user's head jump around back to 0 rotation
                await websocket.send(f"False,{lastSocketString}")
            
            failures = 0
        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                break
            traceback.print_exc()
            failures += 1
            if failures > 30:
                break
        del frame

# Pushes string to port 7010
asyncio.get_event_loop().run_until_complete(
    websockets.serve(facetrack, 'localhost', 7010))
asyncio.get_event_loop().run_forever()