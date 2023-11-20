ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPS_DIR="${ROOT_DIR}/deps"
ISAACGYM_DIR="${DEPS_DIR}/isaacgym"

if [ ! -d $ISAACGYM_DIR ]; then
  echo "$ISAACGYM_DIR does not exist."
  exit
fi

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

eval "$(~/miniconda3/bin/conda shell.bash hook)"


conda env create -f environment.yml

conda activate mpd

if [[ $OSTYPE == 'darwin'* ]]; then
#    pip install torch torchvision
    conda install pytorch torchvision -c pytorch
else
#    pip install torch==2.0.0 torchvision==0.15.0
#    conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
     pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
fi

conda env config vars set CUDA_HOME=""

echo "-------> Installing experiment_launcher"
cd ${DEPS_DIR}/experiment_launcher            && pip install -e .
echo "-------> Installing torch_robotics"
cd ${DEPS_DIR}/torch_robotics                 && pip install -e .
echo "-------> Installing motion_planning_baselines"
cd ${DEPS_DIR}/motion_planning_baselines      && pip install -e .
echo "-------> Installing isaacgym"
cd ${DEPS_DIR}/isaacgym/python                && pip install -e .
echo "-------> Installing storm"
cd ${DEPS_DIR}/storm                          && pip install -e .

echo "-------> Installing this library"
cd ${ROOT_DIR} && pip install -e .

# ncurses is causing an error using the linux command watch, htop, ...
conda remove --force ncurses --yes

conda install -c "conda-forge/label/cf202003" gdown --yes
pip install --upgrade --no-cache-dir gdown

