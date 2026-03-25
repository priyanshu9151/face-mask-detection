import cv2
import numpy as np

def detect_mask_improved():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Focus on nose and mouth area (more precise)
            nose_mouth_region = frame[y+int(h*0.45):y+int(h*0.85), x+int(w*0.25):x+int(w*0.75)]
            
            if nose_mouth_region.size == 0:
                continue
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(nose_mouth_region, cv2.COLOR_BGR2HSV)
            
            # Multiple mask color ranges
            # Blue masks
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # White/light masks
            white_lower = np.array([0, 0, 180])
            white_upper = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # Black/dark masks
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([180, 255, 80])
            black_mask = cv2.inRange(hsv, black_lower, black_upper)
            
            # Green masks
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Combine all mask colors
            combined_mask = cv2.bitwise_or(blue_mask, white_mask)
            combined_mask = cv2.bitwise_or(combined_mask, black_mask)
            combined_mask = cv2.bitwise_or(combined_mask, green_mask)
            
            # Calculate mask coverage
            mask_pixels = cv2.countNonZero(combined_mask)
            total_pixels = nose_mouth_region.shape[0] * nose_mouth_region.shape[1]
            coverage = mask_pixels / total_pixels
            
            # Edge detection for texture analysis
            gray_region = cv2.cvtColor(nose_mouth_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = cv2.countNonZero(edges) / total_pixels
            
            # Brightness analysis
            brightness = np.mean(gray_region)
            
            # Advanced detection with multiple checks
            mask_detected = False
            
            # Check for skin color in the region (no mask should show skin)
            skin_lower = np.array([0, 20, 70])
            skin_upper = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            skin_ratio = cv2.countNonZero(skin_mask) / total_pixels
            
            # Color variance (masks are more uniform than skin)
            color_variance = np.var(gray_region)
            
            # Multiple criteria for mask detection
            score = 0
            
            # High mask color coverage
            if coverage > 0.3:
                score += 3
            elif coverage > 0.2:
                score += 2
            elif coverage > 0.15:
                score += 1
            
            # Low skin visibility (covered by mask)
            if skin_ratio < 0.1:
                score += 2
            elif skin_ratio < 0.2:
                score += 1
            
            # Uniform color (typical of masks)
            if color_variance < 400:
                score += 2
            elif color_variance < 600:
                score += 1
            
            # Strong edges (fabric texture)
            if edge_density > 0.15:
                score += 2
            elif edge_density > 0.1:
                score += 1
            
            # Dark region (shadow from mask)
            if brightness < 90:
                score += 2
            elif brightness < 110:
                score += 1
            
            # Need at least 4 points to detect mask
            if score >= 4:
                mask_detected = True
            
            # Final decision
            if mask_detected:
                label = "Mask Detected"
                color = (0, 255, 0)  # Green
            else:
                label = "No Mask Detected"
                color = (0, 0, 255)  # Red
            
            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            

        
        cv2.imshow('Improved Face Mask Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting improved face mask detection...")
    print("Press 'q' to quit")
    detect_mask_improved()