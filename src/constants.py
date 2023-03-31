from pathlib import Path
from tensorflow.data import AUTOTUNE
import tensorflow as tf

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
interim_data_dir = data_dir / 'interim'
processed_data_dir = data_dir / 'processed'

models_dir = project_dir / 'models' / 'current'
results_dir = project_dir / 'results' / 'current'
images_dir = project_dir / 'images' / 'current'

basic_results_dir = results_dir / 'basic'
basic_models_dir = models_dir / 'basic'
basic_images_dir = images_dir / 'basic'
    
if not basic_results_dir.exists():
    basic_results_dir.mkdir(parents = True)
if not basic_models_dir.exists():
    basic_models_dir.mkdir(parents = True)
if not basic_images_dir.exists():
    basic_images_dir.mkdir(parents = True)

cont_results_dir = results_dir / 'cont'
cont_models_dir = models_dir / 'cont'
cont_images_dir = images_dir / 'cont'

if not cont_results_dir.exists():
    cont_results_dir.mkdir(parents = True)
if not cont_models_dir.exists():
    cont_models_dir.mkdir(parents = True)
if not cont_images_dir.exists():
    cont_images_dir.mkdir(parents = True)

fewshot_results_dir = results_dir / 'fewshot'
fewshot_models_dir = models_dir / 'fewshot'
fewshot_images_dir = images_dir / 'fewshot'

if not fewshot_results_dir.exists():
    fewshot_results_dir.mkdir(parents = True)
if not fewshot_models_dir.exists():
    fewshot_models_dir.mkdir(parents = True)
if not fewshot_images_dir.exists():
    fewshot_images_dir.mkdir(parents = True)

paddings = tf.constant([[0,1]])
autotune = AUTOTUNE
img_height = 224
img_width = 224
num_basic_classes = 6
basic_batch_size = 32
cont_batch_size = 32
basic_frozen_layers = 86
cont_frozen_layers = 154
patience = 100
epochs = 1000
basic_lr = 1e-4
basic_finetuning_lr = 1e-6
cont_lr = 1e-5
cont_finetuning_lr = 1e-7