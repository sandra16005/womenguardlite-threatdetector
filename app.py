import cv2
import time
from image_enhancer import enhance_image
from detector import WeaponDetector
from privacy import FaceBlurrer
from twilio.rest import Client

# Twilio credentials (replace with environment variables in production)
ACCOUNT_SID = 'sid'
AUTH_TOKEN = 'token'
TWILIO_NUMBER = 'number'
RECIPIENT_NUMBER = 'rnumber'

# Flag to track if call has been made (to avoid sending multiple alerts)
call_made = False

def send_voice_call(message):
    """Sends a voice call using Twilio"""
    global call_made
    if call_made:
        return 
    
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        call = client.calls.create(
            twiml=f'<Response><Say>{message}</Say></Response>',
            to=RECIPIENT_NUMBER,
            from_=TWILIO_NUMBER
        )
        print(f"Voice call initiated to {RECIPIENT_NUMBER}")
        call_made = True  # Set flag to prevent duplicate calls
    except Exception as e:
        print(f"Failed to initiate voice call: {e}")

def process_frame(frame, detector, blurrer):
    """Process a single frame"""
    start_time = time.time()
    
    # Enhance image
    enhanced_frame = enhance_image(frame)
    
    # Detect weapons
    weapons = detector.detect(enhanced_frame)
    
    # Mark detected weapons and send voice call alert if a threat is detected
    if weapons:
        for weapon in weapons:
            bbox = weapon['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(enhanced_frame, f"Weapon: {weapon['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Blur faces for privacy protection
        enhanced_frame = blurrer.blur_faces(enhanced_frame)
        cv2.putText(enhanced_frame, "THREAT DETECTED!", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Send voice call alert
        send_voice_call("Threat detected! Please check your surroundings immediately.")
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    return enhanced_frame, processing_time

def main():
    print("Starting WomenGuard Lite...")
    
    # Initialize components
    detector = WeaponDetector()
    blurrer = FaceBlurrer()
    
    # Reset call made flag
    global call_made
    call_made = False
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened successfully. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Process frame sequentially
        enhanced_frame, processing_time = process_frame(frame, detector, blurrer)
        
        # Display the resulting frame
        try:
            cv2.imshow('WomenGuard Lite', enhanced_frame)
        except cv2.error as e:
            print(f"Error displaying frame: {e}")
        
        frame_count += 1
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    if frame_count > 0:
        fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({fps:.2f} FPS)")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
