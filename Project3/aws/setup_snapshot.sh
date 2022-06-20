# Setup environment
source activate pytorch_latest_p37
# Requirements are already installed on snapshot
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Download dataset if necessary
if [ ! -d "/home/ubuntu/dataset" ]; then
  echo "Download data..."
  aws s3 sync s3://dlad-ex3-2021/ /home/ubuntu/dataset/
fi
