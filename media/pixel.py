import cv2

video_path = "sample2.MP4"
cap = cv2.VideoCapture(video_path)

def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Klik kiri
        print(f"Koordinat: ({x}, {y})")

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", get_xy)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):  # Tekan 'q' buat keluar
        break

cap.release()
cv2.destroyAllWindows()

# Koordinat: (423, 313)
# Koordinat: (423, 167)