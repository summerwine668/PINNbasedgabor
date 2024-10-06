Reproducible material for Physics-informed neural wavefields with Gabor basis functions - **Tariq Alkhalifah, Xinquan Huang**


# Project structure
This repository is organized as follows:
* :open_file_folder: **gabor2d**: python library containing the training and testing pipeline;
* :open_file_folder: **model_zoo**: python library containing the model architecture;
* :open_file_folder: **utlis**: python library containing the visualization tools and other utils.
* :open_file_folder: **data**: folder containing data;
* :open_file_folder: **conf**: python library containing the configuration;

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `pinngabor.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate pinngabor
```

## Scripts
Run
```
bash run.sh
```
Before running, you can download the data from [Click here](https://drive.google.com/file/d/1UTEm6M2Ex0eeyyVoyHmQ39IduNR5NNvL/view?usp=sharing). After running, go to folder `exp/results/tb` in the root_path produced by the procedures, and you could use `tensorboard` to visualize the trainig process and predictions.

#### Check the results
After finish the training, you could go to the `<run_root>/results/tb` to use `tensorboard --logdir=./` to check the training metrics and testing results.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A6000 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
```bibtex
@article{alkhalifah2023physics,
  title={Physics-informed neural wavefields with Gabor basis functions},
  author={Alkhalifah, Tariq and Huang, Xinquan},
  journal={arXiv preprint arXiv:2310.10602},
  year={2023}
}
```
