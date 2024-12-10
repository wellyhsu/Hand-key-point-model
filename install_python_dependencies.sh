#!/bin/bash

# 在任何命令失敗時停止執行
set -e

# 建立conda python=3.9虛擬環境並啟用
conda create -n count_params python=3.9 -y
conda activate count_params

# 安裝torch相關套件
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 安裝torch_geometric相關套件
pip install torch_geometric==2.0.4
pip install pyg_lib torch_scatter torch_sparse==0.6.13 torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html && \

# 安裝其他套件
pip install torchinfo==1.8.0 pandas==1.4.2 openmesh==1.2.1 \
            chumpy==0.70 attrs==21.4.0 brotlipy==0.7.0 certifi==2022.5.18.1 cycler==0.11.0 \
            fonttools==4.33.3 fvcore==0.1.5.post20220512 h5py==3.11.0 imageio==2.19.2 \
            iniconfig==1.1.1 iopath==0.1.9 Jinja2==3.1.2 joblib==1.1.0 kiwisolver==1.4.2 \
            MarkupSafe==2.1.1 matplotlib==3.9.2 mkl-fft==1.3.8 mkl-service==2.4.0 \
            networkx==2.8.1 numpy==1.26.4 opencv-python packaging==21.3 Cython==3.0.11\
            Pillow==10.4.0 pluggy==1.0.0 portalocker==2.4.0 protobuf==3.20.1 openpyxl\
            pycocotools==2.0.8 PyOpenGL==3.1.7 pyparsing==3.0.9 pytest==7.1.2 \
            python-dateutil==2.8.2 pytz==2022.1 PyWavelets==1.3.0 PyYAML==6.0.2 \
            pyzmq==26.2.0 scikit-image==0.19.2 scikit-learn==1.1.1 scipy==1.13.1 \
            tabulate==0.8.9 tensorboardX==2.5 termcolor==1.1.0 threadpoolctl==3.1.0 \
            tifffile==2022.5.4 tomli==2.0.1 tqdm==4.64.0 transforms3d==0.4.2 \
            trimesh==3.12.3 vctoolkit==0.1.9.23 vctools==0.1.6.1 yacs==0.1.8

# 更新 setuptools
pip install --upgrade setuptools==70.0.0


