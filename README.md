# Stylized Interactive Neural Painting  [![Read the Thesis](https://img.shields.io/badge/Read%20the%20Thesis-PDF-blue)](https://github.com/MatteoZanella/Sty-Paint/blob/main/Master%20Thesis%20-%20Stylized%20Interactive%20Neural%20Painting.pdf)

Artificial Intelligence advancements have significantly improved performance in Computer Vision tasks, especially in Neural Painting and Style Transfer. The **Interactive Neural Painting (INP)** task has emerged as a powerful model for human-machine collaboration, where artists can use AI to enhance their creative process by generating realistic artwork from brushstrokes. However, the integration of style transfer into this process opens up new possibilities for AI-assisted painting. This work introduces the **Stylized Interactive Neural Painting (Stylized INP)** task, where the model not only assists in completing a painting but also applies the style of a reference artwork, creating a unique blend of content and style. The **Sty-Paint** model, based on the **I-Paint** framework, is proposed to solve the Stylized INP task, achieving interesting stylization capabilities.



## Contributions and Highlights

The Sty-Paint model, the key development in this work, builds upon the I-Paint model framework (Interactive Neural Painting), integrating Style Transfer methods to produce more artistic and personalized stroke suggestions. Key innovations include:

1. Stylized Transformer: The Context Encoder has been modified to accept also a style reference image as input.
2. EFDM+CT Method: Since the process of fully generataing the target stylized stokes was infeasible with the available hardware, I proposed a faster, approximate alternative to generate stylized strokes sequences for training.
3. Fast Stroke Renderer: The large size of styles × images pairs prompted me to develop an optimized neural renderer with reducd GPU memory usage and higher rendering speed for online sample generation during the training.
4. VGG-based Style Loss: A new training objective that enhances stylization contributions, improving stroke suggestions.

This work also introduces new quantitative evaluation metrics for style transfer performance, including Style Contribution, Style Accuracy, and Style Stroke Distance, providing valuable insights into the model's effectiveness in stylized neural painting tasks.

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
