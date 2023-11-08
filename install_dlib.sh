apt-get update -y
apt-get install -y cmake libsm6 libxext6 libxrender-dev python3-pip

pip3 install scikit-build

apt install -y software-properties-common
apt update -y; apt install -y gcc-6 g++-6

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50

git clone https://github.com/davisking/dlib.git
cd dlib/
python3 setup.py install

python3 -m pip install --upgrade pip

pip3 install onnx onnxruntime onnx-tf onnx-simplifier
pip3 install face_recognition
pip3 install requests futures
pip3 install opencv-python