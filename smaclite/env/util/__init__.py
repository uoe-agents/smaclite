import numpy as np


def point_inside_circle(point: np.ndarray,
                        center: np.ndarray,
                        radius: float) -> bool:
    dpos = np.abs(point - center)
    if np.any(dpos > radius):
        return False
    if dpos.sum() <= radius:
        return True
    return np.inner(dpos, dpos) <= radius ** 2
