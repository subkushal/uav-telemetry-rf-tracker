import cv2

def scan_for_optical_hardware():
    print("[System] Scanning for attached video devices...")
    available_cameras = []
    
    # Check the first 5 possible USB video indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[Success] Found active video feed on source ID: {i}")
            available_cameras.append(i)
            
            # Read one frame to see what it looks like
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Testing Source {i} (Press Space to close)", frame)
                cv2.waitKey(0) # Waits for you to press a key
                
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"[Failed] No device on source ID: {i}")
            
    print(f"\n[System] Scan complete. Available IDs: {available_cameras}")
    print("Update your DroneVideoFeed(source=X) with the correct ID.")

if __name__ == "__main__":
    scan_for_optical_hardware()