# coding: utf-8
import dlib
import openface
import numpy as np
import urllib
import cv2

align = openface.AlignDlib('predictor.dat')
net = openface.TorchNeuralNet('nn4.small2.v1.t7', 96)
detector = dlib.get_frontal_face_detector()


def download_image(url):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def face_detect(url, detection_type=1):
    image = download_image(url)
    faces, _, _ = detector.run(image, 1, detection_type)
    faces = max(faces, key=lambda rect: rect.width() * rect.height())
    return faces, image


def compare_images(image1, box1, image2, box2, threshold=0.99, strong_detection=False):
    alignedFace1 = align.align(96, image1, box1, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    alignedFace2 = align.align(96, image2, box2, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)        

    predicted1 = net.forward(alignedFace1)
    predicted2 = net.forward(alignedFace2)

    distance = predicted1 - predicted2
    distance = np.dot(distance, distance)
    print("Distance is {:.4f}".format(distance))

    return float(distance) < threshold
