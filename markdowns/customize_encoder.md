##### Copyright 2020 The TensorFlow Authors.


```python
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

# Customizing a Transformer Encoder

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/official_models/nlp/customize_encoder"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/models/blob/master/official/colab/nlp/customize_encoder.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/models/blob/master/official/colab/nlp/customize_encoder.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/models/official/colab/nlp/customize_encoder.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Learning objectives

The [TensorFlow Models NLP library](https://github.com/tensorflow/models/tree/master/official/nlp/modeling) is a collection of tools for building and training modern high performance natural language models.

The [TransformEncoder](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/encoder_scaffold.py) is the core of this library, and lots of new network architectures are proposed to improve the encoder. In this Colab notebook, we will learn how to customize the encoder to employ new network architectures.

## Install and import

### Install the TensorFlow Model Garden pip package

*  `tf-models-official` is the stable Model Garden package. Note that it may not include the latest changes in the `tensorflow_models` github repo. To include latest changes, you may install `tf-models-nightly`,
which is the nightly Model Garden package created daily automatically.
*  `pip` will install all models and dependencies automatically.


```python
!pip install -q tf-models-official==2.3.0
```

    ERROR: Exception:
    Traceback (most recent call last):
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\urllib3\response.py", line 438, in _error_catcher
        yield
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\urllib3\response.py", line 519, in read
        data = self._fp.read(amt) if not fp_closed else b""
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\cachecontrol\filewrapper.py", line 62, in read
        data = self.__fp.read(amt)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\http\client.py", line 461, in read
        n = self.readinto(b)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\http\client.py", line 505, in readinto
        n = self.fp.readinto(b)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\socket.py", line 589, in readinto
        return self._sock.recv_into(b)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\ssl.py", line 1071, in recv_into
        return self.read(nbytes, buffer)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\ssl.py", line 929, in read
        return self._sslobj.read(len, buffer)
    socket.timeout: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\cli\base_command.py", line 180, in _main
        status = self.run(options, args)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\cli\req_command.py", line 205, in wrapper
        return func(self, options, args)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\commands\install.py", line 319, in run
        reqs, check_supported_wheels=not options.target_dir
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\resolver.py", line 128, in resolve
        requirements, max_rounds=try_to_avoid_resolution_too_deep
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 473, in resolve
        state = resolution.resolve(requirements, max_rounds=max_rounds)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 367, in resolve
        failure_causes = self._attempt_to_pin_criterion(name)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 213, in _attempt_to_pin_criterion
        criteria = self._get_criteria_to_update(candidate)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 203, in _get_criteria_to_update
        name, crit = self._merge_into_criterion(r, parent=candidate)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 172, in _merge_into_criterion
        if not criterion.candidates:
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\resolvelib\structs.py", line 139, in __bool__
        return bool(self._sequence)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 143, in __bool__
        return any(self)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 129, in <genexpr>
        return (c for c in iterator if id(c) not in self._incompatible_ids)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 33, in _iter_built
        candidate = func()
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\factory.py", line 205, in _make_candidate_from_link
        version=version,
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 312, in __init__
        version=version,
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 151, in __init__
        self.dist = self._prepare()
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 234, in _prepare
        dist = self._prepare_distribution()
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 318, in _prepare_distribution
        self._ireq, parallel_builds=True
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\operations\prepare.py", line 508, in prepare_linked_requirement
        return self._prepare_linked_requirement(req, parallel_builds)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\operations\prepare.py", line 552, in _prepare_linked_requirement
        self.download_dir, hashes
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\operations\prepare.py", line 243, in unpack_url
        hashes=hashes,
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\operations\prepare.py", line 102, in get_http_url
        from_path, content_type = download(link, temp_dir.path)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\network\download.py", line 157, in __call__
        for chunk in chunks:
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_internal\network\utils.py", line 86, in response_chunks
        decode_content=False,
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\urllib3\response.py", line 576, in stream
        data = self.read(amt=amt, decode_content=decode_content)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\urllib3\response.py", line 541, in read
        raise IncompleteRead(self._fp_bytes_read, self.length_remaining)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\contextlib.py", line 130, in __exit__
        self.gen.throw(type, value, traceback)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\pip\_vendor\urllib3\response.py", line 443, in _error_catcher
        raise ReadTimeoutError(self._pool, None, "Read timed out.")
    pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
    

### Import Tensorflow and other libraries


```python
import numpy as np
import tensorflow as tf

from official.modeling import activations
from official.nlp import modeling
from official.nlp.modeling import layers, losses, models, networks
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    ~\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow.py in <module>
         57 
    ---> 58   from tensorflow.python.pywrap_tensorflow_internal import *
         59 
    

    ~\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py in <module>
         27             return _mod
    ---> 28     _pywrap_tensorflow_internal = swig_import_helper()
         29     del swig_import_helper
    

    ~\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py in swig_import_helper()
         23             try:
    ---> 24                 _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
         25             finally:
    

    ~\Anaconda3\envs\tensorflow1\lib\imp.py in load_module(name, file, filename, details)
        241         else:
    --> 242             return load_dynamic(name, filename, file)
        243     elif type_ == PKG_DIRECTORY:
    

    ~\Anaconda3\envs\tensorflow1\lib\imp.py in load_dynamic(name, path, file)
        341             name=name, loader=loader, origin=path)
    --> 342         return _load(spec)
        343 
    

    ImportError: DLL load failed: A dynamic link library (DLL) initialization routine failed.

    
    During handling of the above exception, another exception occurred:
    

    ImportError                               Traceback (most recent call last)

    <ipython-input-6-bf4dd5f48d70> in <module>
          1 import numpy as np
    ----> 2 import tensorflow as tf
          3 
          4 from official.modeling import activations
          5 from official.nlp import modeling
    

    ~\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\__init__.py in <module>
         39 import sys as _sys
         40 
    ---> 41 from tensorflow.python.tools import module_util as _module_util
         42 from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
         43 
    

    ~\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\__init__.py in <module>
         48 import numpy as np
         49 
    ---> 50 from tensorflow.python import pywrap_tensorflow
         51 
         52 # Protocol buffers
    

    ~\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow.py in <module>
         67 for some common reasons and solutions.  Include the entire stack trace
         68 above this error message when asking for help.""" % traceback.format_exc()
    ---> 69   raise ImportError(msg)
         70 
         71 # pylint: enable=wildcard-import,g-import-not-at-top,unused-import,line-too-long
    

    ImportError: Traceback (most recent call last):
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
        from tensorflow.python.pywrap_tensorflow_internal import *
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 28, in <module>
        _pywrap_tensorflow_internal = swig_import_helper()
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
        _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\imp.py", line 242, in load_module
        return load_dynamic(name, filename, file)
      File "C:\Users\Apidwalin\Anaconda3\envs\tensorflow1\lib\imp.py", line 342, in load_dynamic
        return _load(spec)
    ImportError: DLL load failed: A dynamic link library (DLL) initialization routine failed.
    
    
    Failed to load the native TensorFlow runtime.
    
    See https://www.tensorflow.org/install/errors
    
    for some common reasons and solutions.  Include the entire stack trace
    above this error message when asking for help.


## Canonical BERT encoder

Before learning how to customize the encoder, let's firstly create a canonical BERT enoder and use it to instantiate a `BertClassifier` for classification task.


```python
cfg = {
    "vocab_size": 100,
    "hidden_size": 32,
    "num_layers": 3,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "activation": activations.gelu,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "sequence_length": 16,
    "type_vocab_size": 2,
    "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
}
bert_encoder = modeling.networks.TransformerEncoder(**cfg)

def build_classifier(bert_encoder):
  return modeling.models.BertClassifier(bert_encoder, num_classes=2)

canonical_classifier_model = build_classifier(bert_encoder)
```

`canonical_classifier_model` can be trained using the training data. For details about how to train the model, please see the colab [fine_tuning_bert.ipynb](https://github.com/tensorflow/models/blob/master/official/colab/fine_tuning_bert.ipynb). We skip the code that trains the model here.

After training, we can apply the model to do prediction.



```python
def predict(model):
  batch_size = 3
  np.random.seed(0)
  word_ids = np.random.randint(
      cfg["vocab_size"], size=(batch_size, cfg["sequence_length"]))
  mask = np.random.randint(2, size=(batch_size, cfg["sequence_length"]))
  type_ids = np.random.randint(
      cfg["type_vocab_size"], size=(batch_size, cfg["sequence_length"]))
  print(model([word_ids, mask, type_ids], training=False))

predict(canonical_classifier_model)
```

## Customize BERT encoder

One BERT encoder consists of an embedding network and multiple transformer blocks, and each transformer block contains an attention layer and a feedforward layer.

We provide easy ways to customize each of those components via (1)
[EncoderScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/encoder_scaffold.py) and (2) [TransformerScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py).

### Use EncoderScaffold

`EncoderScaffold` allows users to provide a custom embedding subnetwork
  (which will replace the standard embedding logic) and/or a custom hidden layer class (which will replace the `Transformer` instantiation in the encoder).

#### Without Customization

Without any customization, `EncoderScaffold` behaves the same the canonical `TransformerEncoder`.

As shown in the following example, `EncoderScaffold` can load `TransformerEncoder`'s weights and output the same values:


```python
default_hidden_cfg = dict(
    num_attention_heads=cfg["num_attention_heads"],
    intermediate_size=cfg["intermediate_size"],
    intermediate_activation=activations.gelu,
    dropout_rate=cfg["dropout_rate"],
    attention_dropout_rate=cfg["attention_dropout_rate"],
    kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
)
default_embedding_cfg = dict(
    vocab_size=cfg["vocab_size"],
    type_vocab_size=cfg["type_vocab_size"],
    hidden_size=cfg["hidden_size"],
    seq_length=cfg["sequence_length"],
    initializer=tf.keras.initializers.TruncatedNormal(0.02),
    dropout_rate=cfg["dropout_rate"],
    max_seq_length=cfg["sequence_length"],
)
default_kwargs = dict(
    hidden_cfg=default_hidden_cfg,
    embedding_cfg=default_embedding_cfg,
    num_hidden_instances=cfg["num_layers"],
    pooled_output_dim=cfg["hidden_size"],
    return_all_layer_outputs=True,
    pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(0.02),
)
encoder_scaffold = modeling.networks.EncoderScaffold(**default_kwargs)
classifier_model_from_encoder_scaffold = build_classifier(encoder_scaffold)
classifier_model_from_encoder_scaffold.set_weights(
    canonical_classifier_model.get_weights())
predict(classifier_model_from_encoder_scaffold)
```

#### Customize Embedding

Next, we show how to use a customized embedding network.

We firstly build an embedding network that will replace the default network. This one will have 2 inputs (`mask` and `word_ids`) instead of 3, and won't use positional embeddings.


```python
word_ids = tf.keras.layers.Input(
    shape=(cfg['sequence_length'],), dtype=tf.int32, name="input_word_ids")
mask = tf.keras.layers.Input(
    shape=(cfg['sequence_length'],), dtype=tf.int32, name="input_mask")
embedding_layer = modeling.layers.OnDeviceEmbedding(
    vocab_size=cfg['vocab_size'],
    embedding_width=cfg['hidden_size'],
    initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
    name="word_embeddings")
word_embeddings = embedding_layer(word_ids)
attention_mask = layers.SelfAttentionMask()([word_embeddings, mask])
new_embedding_network = tf.keras.Model([word_ids, mask],
                                       [word_embeddings, attention_mask])
```

Inspecting `new_embedding_network`, we can see it takes two inputs:
`input_word_ids` and `input_mask`.


```python
tf.keras.utils.plot_model(new_embedding_network, show_shapes=True, dpi=48)
```

We then can build a new encoder using the above `new_embedding_network`.


```python
kwargs = dict(default_kwargs)

# Use new embedding network.
kwargs['embedding_cls'] = new_embedding_network
kwargs['embedding_data'] = embedding_layer.embeddings

encoder_with_customized_embedding = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder_with_customized_embedding)
# ... Train the model ...
print(classifier_model.inputs)

# Assert that there are only two inputs.
assert len(classifier_model.inputs) == 2
```

#### Customized Transformer

User can also override the [hidden_cls](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/encoder_scaffold.py#L103) argument in `EncoderScaffold`'s constructor to employ a customized Transformer layer.

See [ReZeroTransformer](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/rezero_transformer.py) for how to implement a customized Transformer layer.

Following is an example of using `ReZeroTransformer`:



```python
kwargs = dict(default_kwargs)

# Use ReZeroTransformer.
kwargs['hidden_cls'] = modeling.layers.ReZeroTransformer

encoder_with_rezero_transformer = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder_with_rezero_transformer)
# ... Train the model ...
predict(classifier_model)

# Assert that the variable `rezero_alpha` from ReZeroTransformer exists.
assert 'rezero_alpha' in ''.join([x.name for x in classifier_model.trainable_weights])
```

### Use [TransformerScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py)

The above method of customizing `Transformer` requires rewriting the whole `Transformer` layer, while sometimes you may only want to customize either attention layer or feedforward block. In this case, [TransformerScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py) can be used.



#### Customize Attention Layer

User can also override the [attention_cls](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py#L45) argument in `TransformerScaffold`'s constructor to employ a customized Attention layer.

See [TalkingHeadsAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/talking_heads_attention.py) for how to implement a customized `Attention` layer.

Following is an example of using [TalkingHeadsAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/talking_heads_attention.py):


```python
# Use TalkingHeadsAttention
hidden_cfg = dict(default_hidden_cfg)
hidden_cfg['attention_cls'] = modeling.layers.TalkingHeadsAttention

kwargs = dict(default_kwargs)
kwargs['hidden_cls'] = modeling.layers.TransformerScaffold
kwargs['hidden_cfg'] = hidden_cfg

encoder = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder)
# ... Train the model ...
predict(classifier_model)

# Assert that the variable `pre_softmax_weight` from TalkingHeadsAttention exists.
assert 'pre_softmax_weight' in ''.join([x.name for x in classifier_model.trainable_weights])
```

#### Customize Feedforward Layer

Similiarly, one could also customize the feedforward layer.

See [GatedFeedforward](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/gated_feedforward.py) for how to implement a customized feedforward layer.

Following is an example of using [GatedFeedforward](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/gated_feedforward.py).


```python
# Use TalkingHeadsAttention
hidden_cfg = dict(default_hidden_cfg)
hidden_cfg['feedforward_cls'] = modeling.layers.GatedFeedforward

kwargs = dict(default_kwargs)
kwargs['hidden_cls'] = modeling.layers.TransformerScaffold
kwargs['hidden_cfg'] = hidden_cfg

encoder_with_gated_feedforward = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder_with_gated_feedforward)
# ... Train the model ...
predict(classifier_model)

# Assert that the variable `gate` from GatedFeedforward exists.
assert 'gate' in ''.join([x.name for x in classifier_model.trainable_weights])
```

### Build a new Encoder using building blocks from KerasBERT.

Finally, you could also build a new encoder using building blocks in the modeling library.

See [AlbertTransformerEncoder](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/albert_transformer_encoder.py) as an example:



```python
albert_encoder = modeling.networks.AlbertTransformerEncoder(**cfg)
classifier_model = build_classifier(albert_encoder)
# ... Train the model ...
predict(classifier_model)
```

Inspecting the `albert_encoder`, we see it stacks the same `Transformer` layer multiple times.


```python
tf.keras.utils.plot_model(albert_encoder, show_shapes=True, dpi=48)
```
