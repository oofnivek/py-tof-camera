import cv2
import numpy as np
import ArducamDepthCamera as ac

import smbus
import time
import math
import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"tof_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
logging.getLogger().handlers[1].setLevel(logging.INFO)  # keep per-frame debug detail out of the console
log = logging.getLogger("tof")

bus = smbus.SMBus(1)
address = 0x53

bus.write_byte_data(address, 0x2D, 0x08)
bus.write_byte_data(address, 0x31, 0x08)

time.sleep(0.5)

MAX_DISTANCE = 4  # maximum valid distance in meters
scale_percent = 300  # percent of original size
width = int(240 * scale_percent / 100)
height = int(180 * scale_percent / 100)
dim = (width, height)

# Field of view, used when the SDK doesn't report intrinsics (Arducam ToF spec values)
HFOV = 64.3
VFOV = 50.4
OBJECT_MIN_HEIGHT = 1.5  # cm above the floor to count as part of the object
FLOOR_TOLERANCE = 2.0  # cm from the floor level to count as a floor point
MAX_FLOOR_TILT = 15  # degrees allowed between the fitted floor and accelerometer up
ACCEL_SMOOTHING = 0.3  # weight of the newest accelerometer reading (handheld jitter)
DEPTH_EDGE_JUMP = 10.0  # cm, min depth step between neighbouring pixels that separates two surfaces
DEPTH_EDGE_RATIO = 0.25  # the step also has to exceed this fraction of the depth (grazing faces step fast)

class UserRect:
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    @property
    def rect(self):
        return (
            self.start_x,
            self.start_y,
            self.end_x - self.start_x,
            self.end_y - self.start_y,
        )

    @property
    def slice(self):
        return (slice(self.start_y, self.end_y), slice(self.start_x, self.end_x))

    @property
    def empty(self):
        return self.start_x == self.end_x and self.start_y == self.end_y


confidence_value = 15
selectRect, followRect = UserRect(), UserRect()
last_frame = {}  # raw inputs of the latest frame, saved as a snapshot on 's' or a crash


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < confidence_value] = (0, 0, 0)
    return preview


def on_mouse(event, x, y, flags, param):
    global selectRect, followRect

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectRect.start_x = x - 4
        selectRect.start_y = y - 4
        selectRect.end_x = x + 4
        selectRect.end_y = y + 4
    else:
        followRect.start_x = x - 4
        followRect.start_y = y - 4
        followRect.end_x = x + 4
        followRect.end_y = y + 4


def on_confidence_changed(value):
    global confidence_value
    confidence_value = value


def usage(argv0):
    print("Usage: python " + argv0 + " [options]")
    print("Available options are:")
    print(" -d        Choose the video to use")


def read_accel_axis(reg):
    data0 = bus.read_byte_data(address, reg)
    data1 = bus.read_byte_data(address, reg + 1)

    value = ((data1 & 0x03) * 256) + data0

    if value > 511:
        value -= 1024
    return value


def accel_to_camera_up(xAccl, yAccl, zAccl):
    # ADXL345 axes in the camera frame (X right, Y down, Z forward):
    # x = image up, y = image left, z = out the back of the camera
    up = np.array([-yAccl, -xAccl, -zAccl], dtype=np.float64)
    return up / np.linalg.norm(up)


def camera_intrinsics(cam, info):
    fx = (info.width / 2) / math.tan(math.radians(HFOV / 2))
    fy = (info.height / 2) / math.tan(math.radians(VFOV / 2))
    cx, cy = info.width / 2, info.height / 2

    # newer SDKs report calibrated intrinsics (x100); use them if present and sane
    controls = ("INTRINSIC_FX", "INTRINSIC_FY", "INTRINSIC_CX", "INTRINSIC_CY")
    if all(hasattr(ac.TOFControl, c) for c in controls):
        values = [cam.getControl(getattr(ac.TOFControl, c)) / 100 for c in controls]
        if 50 < values[0] < 1000 and 50 < values[1] < 1000:
            fx, fy, cx, cy = values
    return fx, fy, cx, cy


def fit_floor(points, up):
    """Return (n, d) so that n . p + d is a point's height above the floor, or None."""
    heights = points @ up
    bins = np.arange(heights.min(), heights.max() + 1.0, 1.0)
    if len(bins) < 2:
        return None
    counts, edges = np.histogram(heights, bins)

    # the floor is the lowest well-populated level; objects sit above it
    floor_bin = np.argmax(counts >= 0.25 * counts.max())
    floor_level = edges[floor_bin] + 0.5
    floor_pts = points[np.abs(heights - floor_level) < FLOOR_TOLERANCE]
    if len(floor_pts) < 50:
        return None

    centroid = floor_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(floor_pts - centroid, full_matrices=False)
    n = vt[2]
    if n @ up < 0:
        n = -n
    tilt = math.degrees(math.acos(min(1.0, n @ up)))
    if tilt > MAX_FLOOR_TILT:
        log.warning("Floor fit off by %.1f deg from accelerometer up, using accelerometer", tilt)
        n = up  # fit landed on something else (e.g. a wall), trust the accelerometer
    log.debug("Floor: %d points, normal=%s, tilt vs up=%.1f deg, camera height=%.1f cm",
              len(floor_pts), np.round(n, 3), tilt, -n @ centroid)
    return n, -n @ centroid


def save_snapshot(reason, **data):
    path = os.path.join(LOG_DIR, f"snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.npz")
    np.savez_compressed(path, **data)
    log.info("Saved snapshot (%s): %s", reason, path)


def main():
    cam = ac.ArducamCamera()
    cfg_path = None
    black_color = (0, 0, 0)
    white_color = (255, 255, 255)

    ret = 0
    if cfg_path is not None:
        ret = cam.openWithFile(cfg_path, 0)
    else:
        ret = cam.open(ac.TOFConnect.CSI, 0)
    if ret != 0:
        log.error("Failed to open camera. Error code: %s", ret)
        return

    ret = cam.start(ac.TOFOutput.DEPTH)
    if ret != 0:
        log.error("Failed to start camera. Error code: %s", ret)
        cam.close()
        return

    r = cam.getControl(ac.TOFControl.RANGE)

    info = cam.getCameraInfo()
    log.info(f"Camera resolution: {info.width}x{info.height}, device type: {info.device_type}")

    fx, fy, cx, cy = camera_intrinsics(cam, info)
    log.info(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    u, v = np.meshgrid(np.arange(info.width), np.arange(info.height))
    center_dist = (u - info.width / 2) ** 2 + (v - info.height / 2) ** 2
    up = None

    window_name = "Segmented RGB Image"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # cv2.setMouseCallback(window_name, on_mouse)

    if info.device_type == ac.TOFDeviceType.VGA:
        # Only VGA support confidence
        cv2.createTrackbar(
            "confidence", window_name, confidence_value, 255, on_confidence_changed
        )

    while True:
        frame = cam.requestFrame(2000)
        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.getDepthData()
            confidence_buf = frame.getConfidenceData()

            result_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            result_image = getPreviewRGB(result_image, confidence_buf)

            # cv2.imshow("preview_confidence",confidence_buf)

            # cv2.rectangle(result_image, followRect.rect, white_color, 1)

            # if not selectRect.empty:
                # cv2.rectangle(result_image, selectRect.rect, black_color, 2)
                # print(selectRect.slice)
                # print("select Rect distance:", np.mean(depth_buf[selectRect.slice]))

            xAccl = read_accel_axis(0x32)
            yAccl = read_accel_axis(0x34)
            zAccl = read_accel_axis(0x36)

            roll = math.atan2(yAccl, zAccl) * 180 / math.pi
            pitch = 90 + math.atan2(-xAccl, math.sqrt(yAccl * yAccl + zAccl * zAccl)) * 180 / math.pi
            log.info("Roll: %d  Pitch: %d", roll, pitch)

            up_new = accel_to_camera_up(xAccl, yAccl, zAccl)
            if up is None:
                up = up_new
            else:
                up = (1 - ACCEL_SMOOTHING) * up + ACCEL_SMOOTHING * up_new
                up /= np.linalg.norm(up)
            log.debug("Accel raw=(%d, %d, %d) up=%s", xAccl, yAccl, zAccl, np.round(up, 3))

            # copies, since the SDK buffers are reused after releaseFrame
            last_frame.update(
                depth=depth_buf.copy(), confidence=confidence_buf.copy(),
                accel=np.array([xAccl, yAccl, zAccl]), up=up.copy(),
                intrinsics=np.array([fx, fy, cx, cy]), confidence_value=confidence_value,
            )

            # 3D point (cm) for every pixel, camera frame X right, Y down, Z forward
            depth_cm = depth_buf / 10
            cloud = np.dstack(((u - cx) * depth_cm / fx, (v - cy) * depth_cm / fy, depth_cm))
            valid = (depth_buf > 0) & (depth_buf < MAX_DISTANCE * 1000) & (confidence_buf >= confidence_value)

            label = ""
            floor = fit_floor(cloud[valid], up) if np.count_nonzero(valid) >= 50 else None
            if floor is None:
                log.warning("Floor not found (%d valid pixels)", np.count_nonzero(valid))
            else:
                n, d = floor
                height_map = cloud @ n + d

                # cut along depth jumps so the object doesn't merge with the background behind it
                kernel = np.ones((3, 3), np.uint8)
                depth_valid = np.where(valid, depth_cm, 0).astype(np.float32)
                local_range = cv2.dilate(depth_valid, kernel) - cv2.erode(depth_valid, kernel)
                depth_edges = local_range > np.maximum(DEPTH_EDGE_JUMP, DEPTH_EDGE_RATIO * depth_cm)

                obj_mask = (valid & (height_map > OBJECT_MIN_HEIGHT) & ~depth_edges).astype(np.uint8) * 255
                obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=4)
                big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 30)
                big = big[big != 0]  # label 0 is the background

                if big.size:
                    # the object being aimed at: the blob under (or nearest) the image center
                    candidates = np.isin(labels, big)
                    nearest = np.argmin(np.where(candidates, center_dist, np.inf))
                    object_label = labels.flat[nearest]
                    object_mask = (labels == object_label).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    max_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(result_image, [max_contour], -1, (0, 255, 0), 2)

                    x, y, w, l = cv2.boundingRect(max_contour)
                    center_x = x + w // 2
                    center_y = y + l // 2
                    cv2.rectangle(result_image, (x, y), (x+w, y+l), (255, 0, 0), 2)
                    cv2.circle(result_image, (center_x, center_y), 5, (255, 0, 0), 2)
                    log.info("Position: %d %d", center_x, center_y)

                    # object pixels, eroded to drop edge/flying pixels
                    mask = cv2.erode(object_mask, kernel)
                    selected = (mask > 0) & valid & (height_map > OBJECT_MIN_HEIGHT)

                    if np.count_nonzero(selected) >= 20:
                        object_height = np.percentile(height_map[selected], 98)

                        # footprint: object points projected onto the floor plane
                        e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
                        e1 /= np.linalg.norm(e1)
                        e2 = np.cross(n, e1)
                        pts = cloud[selected]
                        footprint = np.column_stack((pts @ e1, pts @ e2)).astype(np.float32)
                        _, (side_a, side_b), _ = cv2.minAreaRect(footprint)
                        object_length, object_width = max(side_a, side_b), min(side_a, side_b)

                        log.info(f"Dimension: {object_length:.1f} cm (L) x {object_width:.1f} cm (W) x {object_height:.1f} cm (H)")
                        label = f"{object_length:.1f} x {object_width:.1f} x {object_height:.1f} cm"

                        dist = np.median(depth_cm[selected])

                        log.info(f"Slant distance: {dist:.2f} cm")
                        log.info(f"Hor distance: {dist * math.cos(pitch / 180 * math.pi):.2f} cm")
                        log.info(f"Vert distance: {dist * math.sin(pitch / 180 * math.pi):.2f} cm")
                    else:
                        log.warning("No valid depth in object")
                else:
                    log.info("No object detected")

            resized_result_image = cv2.resize(result_image, dim, interpolation=cv2.INTER_AREA)
            if label:
                cv2.putText(resized_result_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white_color, 2)
            cv2.imshow(window_name, resized_result_image)

            cam.releaseFrame(frame)
        else:
            log.warning("No depth frame received")

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s") and last_frame:
            save_snapshot("key press", **last_frame)

    cam.stop()
    cam.close()


if __name__ == "__main__":
    log.info("Logging to %s", log_path)
    try:
        main()
    except Exception:
        log.exception("Crashed")
        if last_frame:
            save_snapshot("crash", **last_frame)
        raise
