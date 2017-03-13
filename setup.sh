#!/usr/bin/env bash
workdir = $(pwd) 

git clone https://github.com/cmusatyalab/openface.git
cd openface
python setup.py install
cd - 
wget -nv http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O predictor.dat.bz2
bunzip2 predictor.dat.bz2
wget -nv https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 -O nn4.small2.v1.t7

git clone https://github.com/NVIDIA/caffe.git
cp Makefile caffe/
cd caffe
git checkout 6d723362f0f7fe1aaba7913ebe51cc59b12c0634
make all -j4
make pycaffe

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc
luarocks install dpnn

cd workdir
wget https://www.dropbox.com/s/xm1sl1j7uvggnxr/web_demo.tar.bz2?dl=0
bunzip2 -dk web_demo.tar.bz2
tar xvf web_demo.tar

