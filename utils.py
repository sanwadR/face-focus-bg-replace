import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation
mp_face = mp.solutions.face_detection

class Segmenter:
    def __init__(self, model_selection=1):
        # model_selection=0 for landscape, 1 for general
        self.selfie = mp_selfie.SelfieSegmentation(model_selection=model_selection)

    def mask(self, frame_bgr, threshold=0.6):
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.selfie.process(rgb)
        if res.segmentation_mask is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        m = (res.segmentation_mask > threshold).astype(np.uint8)  # 1=person
        return m

class FaceFinder:
    def __init__(self, model_selection=1, min_detection_confidence=0.6):
        self.face = mp_face.FaceDetection(model_selection=model_selection,
                                          min_detection_confidence=min_detection_confidence)

    def largest_face_bbox(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face.process(rgb)
        if not res.detections:
            return None
        # Pick the detection with largest area
        best = None
        best_area = 0
        for det in res.detections:
            box = det.location_data.relative_bounding_box
            x, y, bw, bh = box.xmin, box.ymin, box.width, box.height
            # Convert to pixel coords
            x1 = max(int(x * w), 0)
            y1 = max(int(y * h), 0)
            x2 = min(int((x + bw) * w), w - 1)
            y2 = min(int((y + bh) * h), h - 1)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        return best

def apply_background(frame_bgr, mask01, mode='blur', bg_bgr=None, blur_ksize=35):
    h, w = frame_bgr.shape[:2]
    mask = mask01.astype(np.uint8)
    mask3 = cv2.merge([mask, mask, mask])

    if mode == 'remove':
        # Transparent background: return BGRA with alpha channel
        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        alpha = (mask * 255).astype(np.uint8)
        bgra[:, :, 3] = alpha
        return bgra

    if mode == 'blur':
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        out = mask3 * frame_bgr + (1 - mask3) * blurred
        return out.astype(np.uint8)

    if mode == 'replace':
        if bg_bgr is None:
            return frame_bgr
        bg_resized = cv2.resize(bg_bgr, (w, h))
        out = mask3 * frame_bgr + (1 - mask3) * bg_resized
        return out.astype(np.uint8)

    return frame_bgr

def crop_to_face(frame_bgr, bbox, margin=0.35):
    if bbox is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    X1 = max(0, x1 - mx)
    Y1 = max(0, y1 - my)
    X2 = min(w, x2 + mx)
    Y2 = min(h, y2 + my)
    return frame_bgr[Y1:Y2, X1:X2]
