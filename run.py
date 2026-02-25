import cv2
import numpy as np
import math

# =========================
# CONFIG
# =========================
pixel_per_mm = 100  # contoh: 100 pixel = 1 mm
CAM_INDEX = 0          # ganti sesuai kamera kamu

# =========================
# LOAD DATA PATTERN
# =========================
data = np.load("pattern_data.npz", allow_pickle=True)

master_kp_pts = data["keypoints"]
master_des = data["descriptors"]
master_shape = tuple(data["img_shape"])
roi_polygons = data["roi_polygons"].tolist() if "roi_polygons" in data else []

master_kp = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in master_kp_pts]

# =========================
# AKAZE DETECTOR
# =========================
akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# =========================
# CAMERA INIT
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Cannot open camera ❌")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame ❌")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = akaze.detectAndCompute(gray, None)

    frame_out = frame.copy()

    if des2 is not None and master_des is not None:
        matches = bf.knnMatch(master_des, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 10:
            src_pts = np.float32([master_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None and len(roi_polygons) > 0:
                for roi_polygon in roi_polygons:
                    roi_pts = np.float32(roi_polygon).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(roi_pts, H)
                    cv2.polylines(frame_out, [np.int32(transformed)], True, (0, 0, 255), 2)

                    n = len(transformed)
                    for i in range(n):
                        pt1 = transformed[i][0]
                        pt2 = transformed[(i + 1) % n][0]
                        dx = pt2[0] - pt1[0]
                        dy = pt2[1] - pt1[1]
                        length_px = math.hypot(dx, dy)
                        length_mm = length_px / pixel_per_mm

                        mid_x = int((pt1[0] + pt2[0]) / 2)
                        mid_y = int((pt1[1] + pt2[1]) / 2)
                        cv2.putText(
                            frame_out,
                            f"{length_mm:.2f} mm",
                            (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

    cv2.imshow("Live Detection with mm", frame_out)
    key = cv2.waitKey(1)
    if key == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
