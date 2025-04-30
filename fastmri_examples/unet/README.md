# U-Net Model for MRI Reconstruction

This directory contains a PyTorch implementation and code for running
pretrained models based on the paper:

[U-Net: Convolutional networks for biomedical image segmentation (O. Ronneberger et al., 2015)](https://doi.org/10.1007/978-3-319-24574-4_28)

which was used as a baseline model in

[fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

as well as the improvements to the baseline for Georgia Tech's CS 7643 Deep Learning, group project.

The following files

- `unet_knee_sc_leaderboard_20201111.py`
- `unet_knee_mc_leaderboard_20201111.py`
- `unet_brain_leaderboard_20201111.py`

contain code and hyperparameters that were used to train the "fastMRI Repo
U-Net" model on the [fastMRI leaderboards][leadlink]. `train_unet_demo.py`
contains a basic demo designed for lower memory consumption.

To start training the model, run:

```bash
python train_unet_demo.py
```

If you run with no arguments, the script will create a `fastmri_dirs.yaml` file
in the root directory that you can use to store paths for your system. You can
also pass options at the command line:

```bash
python train_unet_demo.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --mask_type MASK_TYPE
```

where `CHALLENGE` is either `singlecoil` or `multicoil` and `MASK_TYPE` is
either `random` (for knee) or `equispaced` (for brain). Training logs and
checkpoints are saved in the current working directory by default.

To run the model on test data:

```bash
python train_unet_demo.py \
    --mode test \
    --test_split TESTSPLIT \
    --challenge CHALLENGE \
    --data_path DATA \
    --resume_from_checkpoint MODEL
```

where `MODEL` is the path to the model checkpoint.`TESTSPLIT` should specify
the test split you want to run on - either `test` or `challenge`.

The outputs will be saved to `reconstructions` directory which can be uploaded
for submission.

## Group Project Version

To run training:

```bash
python train_unet.py \
  --mode train \
  --challenge singlecoil \
  --data_path ~/scratch/fastmri_data/singlecoil_knee \
  --default_root_dir unet_demo/roi_gaussian \
  --mask_type random \
  --gpus 2 \
  --strategy ddp \
  --replace_sampler_ddp False \
  --roi-weighting \
  --roi-mask gaussian \
  --roi-margin 0.2 \
  --roi-strength 5.0 \
  --loss-type l2
```
Add a flag for the default_root_dir if you want to separate experiments (have the checkpoints/logs in separate folders). If you don’t include the flag, it will use the default root director.
You can include –roi-weighting to turn it on, and remove the line to turn it off.
You can change loss-type to l2, roi-mask to gaussian or binary, roi-margin is the margin from the border, and the roi-strength is for gaussian decay. Higher roi-strength means greater decay as you move outside from the center. 5.0 is recommended as baseline.

## Pretrained Models

We have pretrained models uploaded on AWS. For an example on how to download
and use them, please see the `run_pretrained_unet_inference.py` script. The
script is a stripped-down version of model creation, `state_dict` downloading
and loading, and model inference. To run the script, type

```bash
python run_pretrained_unet_inference.py \
    --data_path DATA \
    --output_path OUTPUTS \
    --challenge CHALLENGE
```

where in this case `CHALLENGE` is `unet_knee_sc` for the single-coil knee U-Net,
`unet_knee_mc` for the multi-coil knee U-Net, or `unet_brain_mc` for the multi-
coil brain U-Net. The script will download the model and run on your GPU.

## Implementation Notes

The leaderboard model was trained where the `train` split included both the
`train` and `val` splits from the public data.

[leadlink]: https://fastmri.org/leaderboards/