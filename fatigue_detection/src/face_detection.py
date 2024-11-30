import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

    def detect_faces(self, frame):
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect faces
        results = self.face_detection.process(rgb_frame)
        
        # Draw face detections
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw rectangle around face
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                
                # Display confidence score
                cv2.putText(frame, f'Confidence: {int(detection.score[0]*100)}%',
                          (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (0, 255, 0), 2)
        
        return frame

    def run_detection(self):
        # Initialize video capture
        cap = cv2.VideoCapture(0)  # Use 0 for webcam
        
        while True:
            # Read frame from video capture
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            # Process frame
            frame = self.detect_faces(frame)
            
            # Display the frame
            cv2.imshow('Face Detection', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Create and run the detector
if __name__ == "__main__":
    detector = FaceDetector()
    detector.run_detection()