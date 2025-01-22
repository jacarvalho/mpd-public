This code is deprecated.\
Please see the new version in [https://github.com/jacarvalho/mpd-splines](https://github.com/jacarvalho/mpd-splines/).

---

# Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models


Carvalho, J.; Le, A.T.; Baierl, M.; Koert, D.; Peters, J. (2023). **_Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models_**, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

[<img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" />](https://arxiv.org/abs/2308.01557)


<p float="middle">
  <img src="figures/EnvDense2D-RobotPointMass.gif" width="45%" />
  <img src="figures/EnvSpheres3D-RobotPanda.gif" width="45%" />
</p>

---
This repository implements `MPD` - Motion Planning Diffusion -, a method for learning and planning robot motions with
diffusion models, which was presented at IROS 2023.

**NOTES**
- Since the IROS submission we improve the code for better parallelization and improved the baselines.
Hence, the results in the paper are slightly outated, but the qualitivative results are still valid.
- We changed the data generation to use RRT Connect followed by GPMP2, which we found to lead to smoother results.
- The training of the baseline (CVAE) and comparison with traditional planning methods will be added soon. 


If you have any questions please let me know -- [joao@robot-learning.de](mailto:joao@robot-learning.de)

---
## Installation

Pre-requisites:
- Ubuntu 20.04 (maybe works with other OS)
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

Clone this repository with
```bash
cd ~
git clone --recurse-submodules https://github.com/jacarvalho/mpd-public.git
cd mpd-public
```

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) and extract it under `deps/isaacgym`
```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ~/mpd-public/deps/
cd ~/mpd-public/deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
```

Run the bash setup script to install everything.
```
cd ~/mpd-public
bash setup.sh
```

---
## Running the MPD inference

To try out the MPD inference, first download the data and the trained models. 

```bash
conda activate mpd
```

```bash
gdown --id 1mmJAFg6M2I1OozZcyueKp_AP0HHkCq2k
tar -xvf data_trajectories.tar.gz
gdown --id 1I66PJ5QudCqIZ2Xy4P8e-iRBA8-e2zO1
tar -xvf data_trained_models.tar.gz
```

Run the inference script
```bash
cd scripts/inference
python inference.py
```

Comment out the `model-id` variable in `scripts/inference/inference.py` to try out different models
```python
model_id: str = 'EnvDense2D-RobotPointMass'
model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass'
model_id: str = 'EnvSimple2D-RobotPointMass'
model_id: str = 'EnvSpheres3D-RobotPanda'
```

Depending on the task (`model-id`) you might need to change the weights for collision and smoothness (we will provide an "hyperpameter search" soon.)
```python
weight_grad_cost_collision: float = 3e-2
weight_grad_cost_smoothness: float = 1e-2
```

The results will be saved under `data_trained_models/[model_id]/results_inference/`.

---
## Generate data and train from scratch

We recommend running the follwowing in a SLURM cluster.

```bash
conda activate mpd
```

To regenerate the data:
```bash
cd scripts/generate_data
python launch_generate_trajectories.py
```

To train the model:
```bash
cd scripts/train_diffusion
python launch_train_01.py
```





---
## Citation

If you use our work or code base(s), please cite our article:
```latex
@inproceedings{carvalho2023mpd,
  title={Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models},
  author={Carvalho, J. and  Le, A.T. and  Baierl, M. and  Koert, D. and  Peters, J.},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```


---
## Credits

Parts of this work and software were taken and/or inspired from:
- [https://github.com/jannerm/diffuser](https://github.com/jannerm/diffuser)

