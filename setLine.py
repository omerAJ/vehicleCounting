import cv2

# Replace the URL with your RTSP stream URL
RTSP = ""
rtsp_url = f"rtsp://user:pass@{RTSP}"
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: X: ", x, " Y: ", y)

cap = cv2.VideoCapture(rtsp_url)

cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP Stream", (1920, 1080))
cv2.setMouseCallback("RTSP Stream", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("RTSP Stream", frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
