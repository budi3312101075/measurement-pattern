import cv2
import numpy as np

# =========================
# GLOBAL STORAGE
# =========================
current_polygon = []
all_polygons = []

def mouse_callback(event, x, y, flags, param):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread("image.png")
if img is None:
    print("Image not found ❌")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Register ROI")
cv2.setMouseCallback("Register ROI", mouse_callback)

print("""
INSTRUCTION:
Left Click  : Add point
ENTER       : Finish current polygon
U           : Undo last point
C           : Clear current polygon
R           : Reset all polygons
S           : Save & Exit
""")

while True:
    display = img.copy()

    # Draw finished polygons
    for poly in all_polygons:
        cv2.polylines(display, [np.array(poly)], True, (0, 255, 0), 2)

    # Draw current drawing polygon
    if len(current_polygon) > 1:
        cv2.polylines(display, [np.array(current_polygon)], False, (0, 0, 255), 2)

    # Draw points
    for pt in current_polygon:
        cv2.circle(display, pt, 3, (255, 0, 0), -1)

    cv2.imshow("Register ROI", display)
    key = cv2.waitKey(1) & 0xFF

    # ENTER → finish polygon
    if key == 13:
        if len(current_polygon) >= 2:
            all_polygons.append(current_polygon.copy())
            current_polygon.clear()

    # U → undo last point
    elif key == ord('u'):
        if len(current_polygon) > 0:
            current_polygon.pop()

    # C → clear current polygon
    elif key == ord('c'):
        current_polygon.clear()

    # R → reset all
    elif key == ord('r'):
        current_polygon.clear()
        all_polygons.clear()

    # S → save
    elif key == ord('s'):
        if len(current_polygon) >= 2:
            all_polygons.append(current_polygon.copy())
        break

cv2.destroyAllWindows()

# =========================
# FEATURE EXTRACTION
# =========================
akaze = cv2.AKAZE_create()
kp, des = akaze.detectAndCompute(gray, None)

if des is None or len(kp) == 0:
    print("No keypoints/descriptors found ❌")
    exit()

# =========================
# SAVE
# =========================
np.savez(
    "pattern_data.npz",
    keypoints=np.array([k.pt for k in kp]),
    descriptors=des,
    roi_polygons=np.array(all_polygons, dtype=object),
    img_shape=gray.shape
)

print(f"Saved {len(all_polygons)} ROI(s) Successfully 🔥")
