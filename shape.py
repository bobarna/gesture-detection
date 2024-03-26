import numpy as np

def extend_or_bend(landmarks: list) -> list:
    threshold = 0.9

    result = ['e'] * 5

    thumb = np.empty((3, 3))
    for i in range(3):
        thumb[i, :] = np.array([landmarks[i+2].x - landmarks[i+1].x, landmarks[i+2].y - landmarks[i+1].y, landmarks[i+2].z - landmarks[i+1].z])
    for i in range(2):
        if cos_theta(thumb[i+1, :], thumb[i, :]) < threshold:
            result[0] = 'b'
            break

    index = np.empty((3, 3))
    for i in range(3):
        index[i, :] = np.array([landmarks[i+6].x - landmarks[i+5].x, landmarks[i+6].y - landmarks[i+5].y, landmarks[i+6].z - landmarks[i+5].z])
    for i in range(2):
        if cos_theta(index[i+1, :], index[i, :]) < threshold:
            result[1] = 'b'
            break

    middle = np.empty((3, 3))
    for i in range(3):
        middle[i, :] = np.array([landmarks[i+10].x - landmarks[i+9].x, landmarks[i+10].y - landmarks[i+9].y, landmarks[i+10].z - landmarks[i+9].z])
    for i in range(2):
        if cos_theta(middle[i+1, :], middle[i, :]) < threshold:
            result[2] = 'b'
            break

    ring = np.empty((3, 3))
    for i in range(3):
        ring[i, :] = np.array([landmarks[i+14].x - landmarks[i+13].x, landmarks[i+14].y - landmarks[i+13].y, landmarks[i+14].z - landmarks[i+13].z])
    for i in range(2):
        if cos_theta(ring[i+1, :], ring[i, :]) < threshold:
            result[3] = 'b'
            break

    pinky = np.empty((3, 3))
    for i in range(3):
        pinky[i, :] = np.array([landmarks[i+18].x - landmarks[i+17].x, landmarks[i+18].y - landmarks[i+17].y, landmarks[i+18].z - landmarks[i+17].z])
    for i in range(2):
        if cos_theta(pinky[i+1, :], pinky[i, :]) < threshold:
            result[4] = 'b'
            break
    
    return result
    
def mag(v: np.ndarray) -> float:
    v_square = np.square(v)
    v_mag = np.sum(v_square) ** 0.5
    return v_mag

def cos_theta(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = np.sum(v1 * v2) / (mag(v1) * mag(v2))
    return cos
