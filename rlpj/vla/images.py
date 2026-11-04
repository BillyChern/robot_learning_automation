"""Image preparation shared by convert (training data), policy_client and rollout (inference): BGR camera frame ->
RGB uint8 224x224, letterboxed like openpi's resize_with_pad so the stored images equal what openpi would produce."""
import numpy as np

SIZE = 224


def resize_with_pad(rgb: np.ndarray, height: int = SIZE, width: int = SIZE) -> np.ndarray:
    """Scale to fit (aspect kept), pad with zeros, centred; same geometry as openpi_client.image_tools."""
    import cv2
    h, w = rgb.shape[:2]
    if (h, w) == (height, width):
        return rgb
    ratio = max(w / width, h / height)
    rw, rh = int(w / ratio), int(h / ratio)
    interp = cv2.INTER_AREA if ratio > 1 else cv2.INTER_LINEAR
    small = cv2.resize(rgb, (rw, rh), interpolation=interp)
    out = np.zeros((height, width, rgb.shape[2]), rgb.dtype)
    y0, x0 = max(0, int((height - rh) / 2)), max(0, int((width - rw) / 2))
    out[y0:y0 + rh, x0:x0 + rw] = small
    return out


def bgr_to_model(bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Camera BGR frame -> RGB uint8 (size, size, 3) as stored in the dataset and sent to the policy."""
    import cv2
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(resize_with_pad(rgb, size, size)).astype(np.uint8, copy=False)
