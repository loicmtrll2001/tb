import numpy as np
import math

JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
}

def angle_between(p1, p2, p3):
    a = np.array(p1[:2])
    b = np.array(p2[:2])
    c = np.array(p3[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle

def distance(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def extract_features_from_keypoints(keypoints):
    features = []
    #angles des bras et des jambes
    angles = [
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
        ("left_elbow", "left_shoulder", "left_hip"),
        ("right_elbow", "right_shoulder", "right_hip"),
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ]
    for a1, a2, a3 in angles:
        features.append(angle_between(
            keypoints[JOINTS[a1]],
            keypoints[JOINTS[a2]],
            keypoints[JOINTS[a3]]
        ))

    #les distances spécifique
    features.append(distance(keypoints[JOINTS["nose"]], keypoints[JOINTS["left_wrist"]]))
    features.append(distance(keypoints[JOINTS["nose"]], keypoints[JOINTS["right_wrist"]]))
    features.append(distance(keypoints[JOINTS["left_shoulder"]], keypoints[JOINTS["left_wrist"]]))
    features.append(distance(keypoints[JOINTS["right_shoulder"]], keypoints[JOINTS["right_wrist"]]))

    #distance des chevilles
    features.append(distance(keypoints[JOINTS["left_ankle"]], keypoints[JOINTS["right_ankle"]]))

    #hauteur avec axe Y
    for joint in ["left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_shoulder", "right_shoulder"]:
        features.append(keypoints[JOINTS[joint]][1])

    #hauteur moyenne du centre de masse
    center_y = np.mean([
        keypoints[JOINTS["left_shoulder"]][1],
        keypoints[JOINTS["right_shoulder"]][1],
        keypoints[JOINTS["left_hip"]][1],
        keypoints[JOINTS["right_hip"]][1]
    ])
    features.append(center_y)
    return features
