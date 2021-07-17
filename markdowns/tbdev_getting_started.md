##### Copyright 2020 The TensorFlow Authors.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Getting started with [TensorBoard.dev](https://tensorboard.dev)

[TensorBoard.dev](https://tensorboard.dev) is a free, public [TensorBoard](https://tensorflow.org/tensorboard) service that enables you to upload and share your ML experiments with everyone.

This notebook trains a simple model and shows how to upload the logs to TensorBoard.dev. [Preview](https://tensorboard.dev/experiment/rldGbR8rRHeCEbkK61SWTQ).

### Setup and imports

This notebook uses TensorBoard features which are only available for versions >= `2.3.0`.


```
import tensorflow as tf
import datetime
from tensorboard.plugins.hparams import api as hp
```

### Train a simple model and create TensorBoard logs


```
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
```

TensorBoard logs are created during training by passing the [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) and [hyperparameters callbacks](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams) to Keras' Model.fit(). These logs can then be uploaded to TensorBoard.dev.



```
model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
hparams_callback = hp.KerasCallback(log_dir, {
    'num_relu_units': 512,
    'dropout': 0.2
})

model.fit(
    x=x_train, 
    y=y_train, 
    epochs=5, 
    validation_data=(x_test, y_test), 
    callbacks=[tensorboard_callback, hparams_callback])
```

### (Jupyter only)  Authorize TensorBoard.dev

**This step is not necessary in Colab**

This step requires you to auth in your shell console, outside of Jupyter.  In your console, execute the following command.

`tensorboard dev list`

As part of this flow, you will be provided with an authorization code. This code is required to consent to the Terms of Service.

### Upload to TensorBoard.dev

Uploading the TensorBoard logs will give you a URL that can be shared with anyone.

Uploaded TensorBoards are public, so do not upload sensitive data.

The uploader will exit when the entire logdir has uploaded.  (This is what the `--one_shot` flag specifies.)


```
!tensorboard dev upload --logdir ./logs \
  --name "Simple experiment with MNIST" \
  --description "Training results from https://colab.sandbox.google.com/github/tensorflow/tensorboard/blob/master/docs/tbdev_getting_started.ipynb" \
  --one_shot
```

Each individual upload has a unique experiment ID. This means that if you start a new upload with the same directory, you will get a new experiment ID. You can view all your uploaded experiments at https://tensorboard.dev/experiments/. Alternatively, you can list your experiments in the terminal using the following command:
```
tensorboard dev list
```


```
!tensorboard dev list
```

### Screenshots of TensorBoard.dev

This is what it will look like when you navigate to https://tensorboard.dev/experiments/:

![screenshot of TensorBoard.dev experiment list](images/tbdev_experiment_list.png "TensorBoard.dev experiment list screenshot")

This is what it will look like when you navigate to your new experiment on TensorBoard.dev:

![screenshot of TensorBoard.dev experiment dashboard](images/tbdev_getting_started.png "TensorBoard.dev experiment dashboard screenshot")

### Deleting your TensorBoard.dev experiment

To remove an experiment you have uploaded, use the `delete` command and specify the appropriate `experiment_id`.
In the above screenshot, the experiment_id is listed in the bottom left corner: `w1lkBAOrR4eH35Y7Lg1DQQ`.


```
# You must replace YOUR_EXPERIMENT_ID with the value output from the previous
# tensorboard `list` command or `upload` command.  For example
# `tensorboard dev delete --experiment_id pQpJNh00RG2Lf1zOe9BrQA`

## !tensorboard dev delete --experiment_id YOUR_EXPERIMENT_ID_HERE
```
