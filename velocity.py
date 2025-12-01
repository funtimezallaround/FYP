import cv2
import numpy as np

#* CONFIG *
VIDEO = "p25_FREE_UNDER_10s.mp4"
OUTPUT_VIDEO = "swimmer_velocity_annotated.mp4"
# TODO - put correct values 👇👇
fps = 30.0                 # frames per second
f_pixels = 800.0           # focal length (pixels)
Z = 2.0                    # distance to background plane (m)

cap = cv2.VideoCapture(VIDEO)
ret, prev = cap.read()
if not ret:
    raise RuntimeError("Couldn't open video")

h, w = prev.shape[:2]
out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h))

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# detect initial features
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000,
                             qualityLevel=0.01, minDistance=7)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None,
                                           winSize=(21, 21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None:
        break

    good_prev = p0[st == 1]
    good_cur = p1[st == 1]
    flows = good_cur - good_prev

    if len(flows) < 5:
        # re-detect if features are lost
        p0 = cv2.goodFeaturesToTrack(prev_gray, 1000, 0.01, 7)
        prev_gray = frame_gray
        continue

    # median optical flow (pixels/frame)
    median_dx = np.median(flows[:, 0])
    median_dy = np.median(flows[:, 1])

    # convert to real velocity (m/s)
    tx = -median_dx * (Z / f_pixels)
    ty = -median_dy * (Z / f_pixels)
    speed_mps = np.sqrt(tx**2 + ty**2) * fps

    # visualize features and flow
    vis = frame.copy()
    for (x1, y1), (x2, y2) in zip(good_prev, good_cur):
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.circle(vis, (int(x2), int(y2)), 2, (0, 255, 255), -1)

    # draw median background flow arrow
    center = (w // 2, h // 2)
    arrow_end = (int(center[0] + median_dx * 10),
                 int(center[1] + median_dy * 10))
    cv2.arrowedLine(vis, center, arrow_end, (0, 0, 255), 2, tipLength=0.3)

    # overlay velocity
    cv2.putText(vis, f"Estimated speed: {speed_mps:.2f} m/s",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Velocity Estimation", vis)
    out.write(vis)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to stop
        break

    # prepare next
    prev_gray = frame_gray
    p0 = good_cur.reshape(-1, 1, 2)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
