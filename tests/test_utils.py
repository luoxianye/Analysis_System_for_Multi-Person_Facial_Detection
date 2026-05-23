import numpy as np

from utils import crop_face


def test_crop_face_inside_boundary():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = crop_face(img, (10, 10, 40, 40))
    assert face.shape[0] == 40
    assert face.shape[1] == 40


def test_crop_face_outside_boundary():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = crop_face(img, (-10, -10, 50, 50))
    assert face.size > 0
