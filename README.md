# Interactive Neural Painting

### Model
The model implementaion can be found in 
`model/model_type/VAE.py`
the different modules are implemented in `model/networks/`

### Train
To train the model run

`python train.py --config ./configs/train/train_options.yaml`

change the configuration file to try different versions.

### Evaluation
To evaluate the models, run:

`python train.py --config ./configs/eval/eval_oxford.yaml --checkpoint --use-pt --use-snp --use-snp2`
