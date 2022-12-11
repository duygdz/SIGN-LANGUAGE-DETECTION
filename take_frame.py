import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp



mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR -> RGB
    image.flags.writeable = False   #img not writable
    results = model.process(image) #making prediction
    image.flags.writeable = True    #img writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #RGB -> BGR
    return image, results


def draw_styled_landmarks(image, result):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1), #joins
                                mp_drawing.DrawingSpec(color=(255,191,0), thickness=1, circle_radius=1)  #lines
                                ) #draw face connections; 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                                ) #draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,165,0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,191,255), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))

def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #33 landmarks * 4 values each
    pose = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)     
    return np.concatenate([pose, face, lh, rh])

# path for exported data, np array
DATA_PATH = os.path.join('MP_Data')
# list of actions   
actions = np.array(['hello','thanks','salanghaeyo~','sad','family'])
# 30 videos of data
no_sequences = 30
# 30 frames
sequence_length = 30
# Folder start
start_folder = 0

''' for further input
for action in actions: 
    dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
        except:
            pass
'''
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

#collect training data
cap = cv2.VideoCapture(0)  # access webcam
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic: #set percentage of confidence of detection and tracking 
    # Loop through action
    for action in actions:
        #loop through videos
        for sequence in range(start_folder, start_folder+no_sequences):
            #loop through video_length
            for frame_num in range(sequence_length):
                
                # read feed
                ret, frame = cap.read()
                # Detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                #draw landmarks
                draw_styled_landmarks(image,results)


                #collecting data with break
                if frame_num == 0:  #if first frame
                    cv2.putText(image, 'GET READY', (120,200),  #show onscreen "get ready"
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)  #custom font size, color
                    cv2.putText(image, 'Collecting frames for {} takes #{} '.format(action, sequence), (15,12),  #show onscreen current frame
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                     # show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000) # break before start collecting frames
                else:
                    cv2.putText(image, 'Collecting frames for {} takes #{} '.format(action, sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                     # show to screen
                    cv2.imshow('OpenCV Feed', image)


                # EXPORT KEYPOINTS
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
               
                
                #stop condition = 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break        
    #stop camera
    cap.release()
    cv2.destroyAllWindows()  