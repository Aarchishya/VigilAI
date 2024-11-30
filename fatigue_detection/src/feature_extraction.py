import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame  # For audio alerts

class FatigueDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize audio alert
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound('alert.wav')  # You'll need to add an alert sound file
        
        # Eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Mouth landmarks
        self.MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.65
        self.EYE_CLOSED_FRAMES = 0
        self.YAWN_FRAMES = 0
        self.FRAMES_THRESHOLD = 20
        
        # Counters for analytics
        self.yawn_counter = 0
        self.drowsy_counter = 0
        
    def calculate_ear(self, eye_points):
        """Calculate eye aspect ratio"""
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_points):
        """Calculate mouth aspect ratio"""
        A = distance.euclidean(mouth_points[1], mouth_points[5])  # Vertical distance
        C = distance.euclidean(mouth_points[0], mouth_points[4])  # Horizontal distance
        mar = A / C
        return mar
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye points
                left_eye_points = np.array([[face_landmarks.landmark[i].x * frame_width,
                                           face_landmarks.landmark[i].y * frame_height]
                                          for i in self.LEFT_EYE])
                right_eye_points = np.array([[face_landmarks.landmark[i].x * frame_width,
                                            face_landmarks.landmark[i].y * frame_height]
                                           for i in self.RIGHT_EYE])
                
                # Get mouth points
                mouth_points = np.array([[face_landmarks.landmark[i].x * frame_width,
                                        face_landmarks.landmark[i].y * frame_height]
                                       for i in self.MOUTH])
                
                # Calculate ratios
                left_ear = self.calculate_ear(left_eye_points)
                right_ear = self.calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(mouth_points)
                
                # Draw features
                cv2.polylines(frame, [left_eye_points.astype(int)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye_points.astype(int)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [mouth_points.astype(int)], True, (0, 255, 0), 1)
                
                # Display metrics
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check for drowsiness
                if avg_ear < self.EAR_THRESHOLD:
                    self.EYE_CLOSED_FRAMES += 1
                else:
                    self.EYE_CLOSED_FRAMES = 0
                
                # Check for yawning
                if mar > self.MAR_THRESHOLD:
                    self.YAWN_FRAMES += 1
                else:
                    if self.YAWN_FRAMES >= self.FRAMES_THRESHOLD:
                        self.yawn_counter += 1
                    self.YAWN_FRAMES = 0
                
                # Alert conditions
                if self.EYE_CLOSED_FRAMES >= self.FRAMES_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.drowsy_counter += 1
                    self.alert_sound.play()
                
                # if self.YAWN_FRAMES >= self.FRAMES_THRESHOLD:
                #     cv2.putText(frame, "YAWNING!", (10, 120),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                # Display counters
                # cv2.putText(frame, f"Yawns: {self.yawn_counter}", (10, 150),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Drowsy Events: {self.drowsy_counter}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = self.process_frame(frame)
            cv2.imshow('Fatigue Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FatigueDetector()
    detector.run()