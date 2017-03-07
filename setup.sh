#!/usr/bin/env bash
git clone https://github.com/cmusatyalab/openface.git
cd openface
python setup.py install
cd - 
wget -nv http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O predictor.dat.bz2
bunzip2 predictor.dat.bz2
wget -nv https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 -O nn4.small2.v1.t7
