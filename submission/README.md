#watch video to see explanation on how to set up environment
#install python 3.7
#create conda env and install these dependencies 

conda create -n drl python=3.7
conda activate drl

pip install numpy==1.21.6 tensorflow==2.10.1 matplotlib==3.5.3 scikit-learn==1.0.2 seaborn==0.11.2 gym==0.26.2 pandas==1.3.5 scipy==1.7.3 tqdm scapy
python mainHpcc.py