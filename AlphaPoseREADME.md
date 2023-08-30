# Installing AlphaPose on CCV
We are using this [repo](https://github.com/MVIG-SJTU/AlphaPose/tree/master). 

## Before Installing
Load the following modules:
```
module load python/3.9.0
module load cuda/11.7.1
module load gcc/10.2
```

Create a virtual environment and activate it.

Install libraries
```
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Installing
To install:
```
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
python3 setup.py build develop --user
```