# Installing
- Get miniconda https://docs.conda.io/en/latest/miniconda.html
```sh
conda create -n sip-mask python=3.7 -y
conda activate sip-mask-test
conda install -c pytorch torchvision=0.2.1 cudatoolkit=9.0
pip install ninja yacs cython matplotlib tqdm
pip install python-cocoapi
git clone https://github.com/Traderain/SipMask
cd SipMask/Sipmask-VIS
python setup.py develop
```

# Training

- TODO

# Running

- Open Sipmask/Sipmask-VIS in vscode
- Select miniconda3's python
- Press F5 to run the webcam demo