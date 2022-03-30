# Interactive Neural Painting

## Setup

Set up environment and logging before training. 

#### Environment

The environment can be created with conda, run:  
```bash
conda env create -f env.yml  
conda activate inp
```

#### Neural Renderer

Download the `oil-paint brush` pretrained neural renderer from [Stylized Neural Painting](https://github.com/jiupinjia/stylized-neural-painting) (direct [link](https://drive.google.com/file/d/1sqWhgBKqaBJggl2A8sD1bLSq2_B1ScMG/view)).
    
#### Configurations

Modify the path of configuration files:

- `configs/decomposition/painter_config.yaml`:
    * `renderer_checkpoint_dir` path of the pretrained neural renderer.
    * `brush_paths : <your_base_path>/brushstrokes-generation/dataset_acquisition/decomposition/brushes/brush_fromweb2_small_vertical.png`.
    
- `configs/train/train_options.yaml`:
    * `dataset/root` path of the dataset.
    * `dataset/csv_file` path to csv file with train / test split.
    * `renderer/painter_config` path to `painter_config.yaml` file.

#### Logging

Logging is performed on Weight and Biases, login with `wandb login`.

## Dataset
To obtain the Oxford-IIIT Pet INP and ADE 20K Outdoor INP 

1. Download original images from [Oxford Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [ADE20K Outdoor](https://www.kaggle.com/residentmario/ade20k-outdoors).
2. Download parameters files form [here](add_link). For each image in the dataset, it contains the associated brushstrokes decomposition and the re-ordering of the strokes.
3. Render the strokes and create the dataset, run:  
```bash
python -m dataset_acquisition.main_render --dataset_path --strokes_path --images_path --index_path
```

## Training

To train the model run

```bash
python train.py --config ./configs/train/train_options.yaml
```

change the configuration file to try different versions.

## Evaluation

To evaluate the model, run:

```bash
python evaluate_metrics.py --config ./configs/eval/<config_file.yaml> --checkpoint
```

specify the path of the trained model.  
To evaluate baseline models, add the following flags `--use-pt --use-snp --use-snp2`.

## Demo

To try the interactive demo, run:

```bash
python -m demo.paint --img_path --checkpoint
```
specify the image to be painted and the checkpoint path.