import cv2
import numpy as np
import ArducamDepthCamera as ac

import smbus
import time
import math
import logging
import os
from collections import deque
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
RADIAL_DEPTH = True  # depth is distance along each pixel's ray, not along the lens axis
CAMERA_HEIGHT = 90.0  # cm, camera mounting height above the ground
GROUND_SEARCH = 30.0  # cm, look for the ground within this distance of CAMERA_HEIGHT below the camera
OBJECT_MIN_HEIGHT = 5.0  # cm above the floor to count as part of the object (above floor noise)
TOP_FACE_TOLERANCE = 3.0  # cm below the top of the object still counted as its top face
FLOOR_TOLERANCE = 2.0  # cm from the floor plane to count as a floor point
MAX_FLOOR_TILT = 20  # degrees allowed between the fitted floor and accelerometer up (mounting offset)
RANSAC_ITERATIONS = 200  # plane guesses tried when fitting the floor
RANSAC_SAMPLE = 3000  # points used to score each guess
ACCEL_SMOOTHING = 0.1  # weight of the newest accelerometer reading (camera is fixed, so smooth heavily)
DEPTH_FRAMES = 5  # per-pixel median over this many frames to cut ToF noise (box and camera are still)
HEIGHT_COLOR_MAX = 60.0  # cm, height shown as the top (yellow) of the preview colour scale
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
rng = np.random.default_rng()


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
    # x = image up, y = image right, z = forward along the lens (confirmed on the fixed
    # 90 cm mount: tilted down at the box gives x > 0, z < 0)
    up = np.array([yAccl, -xAccl, zAccl], dtype=np.float64)
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
    """Return (n, d) so that n . p + d is a point's height above the ground.

    The ground is the largest roughly level plane the ToF camera sees around CAMERA_HEIGHT
    below the camera (RANSAC), so an accelerometer mounted a few degrees off the lens doesn't
    matter. If too little of it is visible, the configured height and accelerometer are used.
    """
    candidates = points[np.abs(points @ up + CAMERA_HEIGHT) < GROUND_SEARCH]

    if len(candidates) >= 50:
        sample = candidates[rng.choice(len(candidates), min(RANSAC_SAMPLE, len(candidates)), replace=False)]
        min_cos = math.cos(math.radians(MAX_FLOOR_TILT))
        best_count, best_plane = 0, None
        for _ in range(RANSAC_ITERATIONS):
            p0, p1, p2 = sample[rng.choice(len(sample), 3, replace=False)]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n /= norm
            if n @ up < 0:
                n = -n
            if n @ up < min_cos:
                continue  # not level enough to be the ground (e.g. a box side)
            count = np.count_nonzero(np.abs((sample - p0) @ n) < FLOOR_TOLERANCE)
            if count > best_count:
                best_count, best_plane = count, (n, -n @ p0)

        if best_plane is not None:
            # refine on every candidate near the best guess
            n, d = best_plane
            floor_pts = candidates[np.abs(candidates @ n + d) < FLOOR_TOLERANCE]
            centroid = floor_pts.mean(axis=0)
            _, _, vt = np.linalg.svd(floor_pts - centroid, full_matrices=False)
            n = vt[2] if vt[2] @ up > 0 else -vt[2]
            tilt = math.degrees(math.acos(min(1.0, n @ up)))
            log.debug("Floor: %d points, normal=%s, tilt vs up=%.1f deg, camera height=%.1f cm",
                      len(floor_pts), np.round(n, 3), tilt, -n @ centroid)
            return n, -n @ centroid

    log.warning("Ground not visible, using configured camera height %.0f cm", CAMERA_HEIGHT)
    return up, CAMERA_HEIGHT


def median_depth(history):
    """Per-pixel median of the valid readings in history (NaN = invalid); 0 where none are valid."""
    stack = np.stack(history)
    count = np.count_nonzero(~np.isnan(stack), axis=0)
    ordered = np.sort(stack, axis=0)  # NaNs sort to the end
    index = np.maximum((count - 1) // 2, 0)
    median = np.take_along_axis(ordered, index[None], axis=0)[0]
    median[count == 0] = 0
    return median


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
    ray_x, ray_y = (u - cx) / fx, (v - cy) / fy  # each pixel's ray direction at Z = 1
    ray_length = np.sqrt(1 + ray_x ** 2 + ray_y ** 2)
    center_dist = (u - info.width / 2) ** 2 + (v - info.height / 2) ** 2
    up = None
    depth_history = deque(maxlen=DEPTH_FRAMES)

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

            frame_valid = (depth_buf > 0) & (depth_buf < MAX_DISTANCE * 1000) & (confidence_buf >= confidence_value)
            depth_history.append(np.where(frame_valid, depth_buf, np.nan).astype(np.float32))
            depth_mm = median_depth(depth_history)

            # copies, since the SDK buffers are reused after releaseFrame
            last_frame.update(
                depth=depth_mm, depth_raw=depth_buf.copy(), confidence=confidence_buf.copy(),
                accel=np.array([xAccl, yAccl, zAccl]), up=up.copy(),
                intrinsics=np.array([fx, fy, cx, cy]), confidence_value=confidence_value,
            )

            # 3D point (cm) for every pixel, camera frame X right, Y down, Z forward
            depth_cm = depth_mm / 10
            z_cm = depth_cm / ray_length if RADIAL_DEPTH else depth_cm
            cloud = np.dstack((ray_x * z_cm, ray_y * z_cm, z_cm))
            valid = depth_mm > 0

            label = ""
            floor = fit_floor(cloud[valid], up) if np.count_nonzero(valid) >= 50 else None
            if floor is None:
                log.warning("Too few valid pixels (%d)", np.count_nonzero(valid))
            else:
                n, d = floor
                height_map = cloud @ n + d

                # preview: ground in grey, anything above it coloured by height (red low -> yellow high)
                above = valid & (height_map > OBJECT_MIN_HEIGHT)
                height_u8 = (np.clip(height_map / HEIGHT_COLOR_MAX, 0, 1) * 255).astype(np.uint8)
                result_image = np.zeros((*depth_mm.shape, 3), np.uint8)
                result_image[valid] = (128, 128, 128)
                result_image[above] = cv2.applyColorMap(height_u8, cv2.COLORMAP_AUTUMN)[above]

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
                        # top face: the highest layer of the object; height is its level above the ground.
                        # Taken from the uneroded mask: the height band already drops mixed edge pixels
                        top = np.percentile(height_map[selected], 95)
                        top_face = (object_mask > 0) & valid & (height_map > top - TOP_FACE_TOLERANCE)
                        object_height = np.median(height_map[top_face])

                        # length and width: top face projected onto the floor plane (top view)
                        e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
                        e1 /= np.linalg.norm(e1)
                        e2 = np.cross(n, e1)
                        pts = cloud[top_face]
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
