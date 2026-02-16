import cv2
import mediapipe as mp
import time
import pygame
import numpy as np
from pathlib import Path
from collections import deque
from mediapipe.python.solutions import face_mesh

def draw_warning(frame, text="lock in twin"):
    h, w = frame.shape[:2]
   
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h
 
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)    
    
    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (80, 255, 160) , 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160) , 2)
  
    cv2.putText(
        frame,
        text.upper(),
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
def get_head_pose(shape, img_h, img_w):
    face_3d = []
    face_2d = []
    
    points_idx = [1, 199, 33, 263, 61, 291]

    for idx in points_idx:
        lm = shape[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    rmat, jac = cv2.Rodrigues(rot_vec)
    
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360 
    y = angles[1] * 360 
    
    return x, y

def main():
   
    timer = 1.0
    
   
    head_pitch_limit = 0.0 
    
    SMOOTHING_FRAMES = 10
    # -------------------------------------
    
    video_path = Path("skyrim-skeleton.mp4").resolve()
    audio_path = Path("sound.mp3").resolve()
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    pygame.mixer.init()
    if audio_path.exists():
        pygame.mixer.music.load(str(audio_path))
    
    skeleton_cap = cv2.VideoCapture(str(video_path))
    
    face_mesh_landmarks = face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cam.isOpened():
        print("Could not open webcam")
        return
    
    doomscroll = None
    video_playing = False
    
    pitch_history = deque(maxlen=SMOOTHING_FRAMES)

    print("Program Started! Negative Pitch Detection ON.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed_image = face_mesh_landmarks.process(rgb_frame)
        face_landmark_points = processed_image.multi_face_landmarks

        current = time.time()
        is_doomscrolling = False 

        if face_landmark_points:
            landmarks = face_landmark_points[0].landmark
            
            pitch, yaw = get_head_pose(landmarks, height, width)
            
            pitch_history.append(pitch)
            smooth_pitch = sum(pitch_history) / len(pitch_history)
                      
            x_min = int(min([lm.x for lm in landmarks]) * width)
            y_min = int(min([lm.y for lm in landmarks]) * height)
            x_max = int(max([lm.x for lm in landmarks]) * width)
            y_max = int(max([lm.y for lm in landmarks]) * height)

            box_color = (0, 255, 0) if not is_doomscrolling else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                  
            if smooth_pitch < head_pitch_limit:
                is_doomscrolling = True
            
          
            color = (0, 255, 0)
            status = "Safe"
            
            if is_doomscrolling:
                color = (0, 0, 255)
                status = "DOOMSCROLLING!"

            cv2.putText(frame, f"Head Pitch: {smooth_pitch:.1f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Limit: < {head_pitch_limit} | Status: {status}", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if is_doomscrolling:
                if doomscroll is None:
                    doomscroll = current

                if (current - doomscroll) >= timer:
                    if not video_playing:
                        video_playing = True
                        if audio_path.exists():
                            pygame.mixer.music.play(-1)
            else:
                doomscroll = None
                if video_playing:
                    video_playing = False
                    pygame.mixer.music.stop()
        else:
            doomscroll = None
            if video_playing:
                video_playing = False
                pygame.mixer.music.stop()

        if video_playing:
            draw_warning(frame, "STOP DOOMSCROLLING!")
            
            v_ret, v_frame = skeleton_cap.read()
            if not v_ret:
                skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                v_ret, v_frame = skeleton_cap.read()
            
            if v_ret:
                v_frame = cv2.resize(v_frame, (500, 350))
                cv2.imshow('SKELETON WARNING', v_frame)
                cv2.moveWindow('SKELETON WARNING', 50, 100)
        else:
            try:
                cv2.destroyWindow('SKELETON WARNING')
            except:
                pass

        small_frame = cv2.resize(frame, (640, 360))
        cv2.imshow('Doomscroll Detector (Head Only)', small_frame)
        cv2.moveWindow('Doomscroll Detector (Head Only)', 700, 100)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    skeleton_cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == '__main__':
    main()