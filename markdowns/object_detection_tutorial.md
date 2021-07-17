# Object Detection API Demo

<table align="left"><td>
  <a target="_blank"  href="https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab
  </a>
</td><td>
  <a target="_blank"  href="https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb">
    <img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
</td></table>

Welcome to the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image.

> **Important**: This tutorial is to help you through the first step towards using [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to build models. If you just just need an off the shelf model that does the job, see the [TFHub object detection example](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb).

# Setup

Important: If you're running on a local machine, be sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). This notebook includes only what's necessary to run in Colab.

### Install


```python
!pip install -U --pre tensorflow=="2.*"
!pip install tf_slim
```

    Collecting tensorflow==2.*
      Downloading tensorflow-2.6.0rc1-cp37-cp37m-win_amd64.whl (423.2 MB)
    

    ERROR: Exception:
    Traceback (most recent call last):
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\urllib3\response.py", line 438, in _error_catcher
        yield
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\urllib3\response.py", line 519, in read
        data = self._fp.read(amt) if not fp_closed else b""
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\cachecontrol\filewrapper.py", line 62, in read
        data = self.__fp.read(amt)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\http\client.py", line 457, in read
        n = self.readinto(b)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\http\client.py", line 501, in readinto
        n = self.fp.readinto(b)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\socket.py", line 589, in readinto
        return self._sock.recv_into(b)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\ssl.py", line 1071, in recv_into
        return self.read(nbytes, buffer)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\ssl.py", line 929, in read
        return self._sslobj.read(len, buffer)
    socket.timeout: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\cli\base_command.py", line 180, in _main
        status = self.run(options, args)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\cli\req_command.py", line 205, in wrapper
        return func(self, options, args)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\commands\install.py", line 319, in run
        reqs, check_supported_wheels=not options.target_dir
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\resolver.py", line 128, in resolve
        requirements, max_rounds=try_to_avoid_resolution_too_deep
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 473, in resolve
        state = resolution.resolve(requirements, max_rounds=max_rounds)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 341, in resolve
        name, crit = self._merge_into_criterion(r, parent=None)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 172, in _merge_into_criterion
        if not criterion.candidates:
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\resolvelib\structs.py", line 139, in __bool__
        return bool(self._sequence)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 143, in __bool__
        return any(self)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 129, in <genexpr>
        return (c for c in iterator if id(c) not in self._incompatible_ids)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 33, in _iter_built
        candidate = func()
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\factory.py", line 205, in _make_candidate_from_link
        version=version,
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 312, in __init__
        version=version,
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 151, in __init__
        self.dist = self._prepare()
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 234, in _prepare
        dist = self._prepare_distribution()
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 318, in _prepare_distribution
        self._ireq, parallel_builds=True
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\operations\prepare.py", line 508, in prepare_linked_requirement
        return self._prepare_linked_requirement(req, parallel_builds)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\operations\prepare.py", line 552, in _prepare_linked_requirement
        self.download_dir, hashes
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\operations\prepare.py", line 243, in unpack_url
        hashes=hashes,
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\operations\prepare.py", line 102, in get_http_url
        from_path, content_type = download(link, temp_dir.path)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\network\download.py", line 157, in __call__
        for chunk in chunks:
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\cli\progress_bars.py", line 152, in iter
        for x in it:
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_internal\network\utils.py", line 86, in response_chunks
        decode_content=False,
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\urllib3\response.py", line 576, in stream
        data = self.read(amt=amt, decode_content=decode_content)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\urllib3\response.py", line 541, in read
        raise IncompleteRead(self._fp_bytes_read, self.length_remaining)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\contextlib.py", line 130, in __exit__
        self.gen.throw(type, value, traceback)
      File "c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\pip\_vendor\urllib3\response.py", line 443, in _error_catcher
        raise ReadTimeoutError(self._pool, None, "Read timed out.")
    pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
    

    Collecting tf_slim
      Using cached tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
    Requirement already satisfied: absl-py>=0.2.2 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from tf_slim) (0.9.0)
    Requirement already satisfied: six in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from absl-py>=0.2.2->tf_slim) (1.15.0)
    Installing collected packages: tf-slim
    Successfully installed tf-slim-1.1.0
    

Make sure you have `pycocotools` installed


```python
!pip install pycocotools
```

    Collecting pycocotools
      Using cached pycocotools-2.0.2.tar.gz (23 kB)
    Requirement already satisfied: setuptools>=18.0 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from pycocotools) (57.2.0)
    Requirement already satisfied: cython>=0.27.3 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from pycocotools) (0.29.24)
    Requirement already satisfied: matplotlib>=2.1.0 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\matplotlib-3.4.2-py3.7-win-amd64.egg (from pycocotools) (3.4.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\cycler-0.10.0-py3.7.egg (from matplotlib>=2.1.0->pycocotools) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\kiwisolver-1.3.1-py3.7-win-amd64.egg (from matplotlib>=2.1.0->pycocotools) (1.3.1)
    Requirement already satisfied: numpy>=1.16 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from matplotlib>=2.1.0->pycocotools) (1.19.1)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from matplotlib>=2.1.0->pycocotools) (8.3.1)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from matplotlib>=2.1.0->pycocotools) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from matplotlib>=2.1.0->pycocotools) (2.8.2)
    Requirement already satisfied: six in c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages (from cycler>=0.10->matplotlib>=2.1.0->pycocotools) (1.15.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py): started
      Building wheel for pycocotools (setup.py): finished with status 'error'
      Running setup.py clean for pycocotools
    Failed to build pycocotools
    Installing collected packages: pycocotools
        Running setup.py install for pycocotools: started
        Running setup.py install for pycocotools: finished with status 'error'
    

      ERROR: Command errored out with exit status 1:
       command: 'c:\users\apidwalin\anaconda3\envs\tf\python.exe' -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\APIDWA~1\\AppData\\Local\\Temp\\pip-install-jt0cdwch\\pycocotools_ad600866e39342aa97877c453bcad251\\setup.py'"'"'; __file__='"'"'C:\\Users\\APIDWA~1\\AppData\\Local\\Temp\\pip-install-jt0cdwch\\pycocotools_ad600866e39342aa97877c453bcad251\\setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d 'C:\Users\APIDWA~1\AppData\Local\Temp\pip-wheel-slx_95l4'
           cwd: C:\Users\APIDWA~1\AppData\Local\Temp\pip-install-jt0cdwch\pycocotools_ad600866e39342aa97877c453bcad251\
      Complete output (16 lines):
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build\lib.win-amd64-3.7
      creating build\lib.win-amd64-3.7\pycocotools
      copying pycocotools\coco.py -> build\lib.win-amd64-3.7\pycocotools
      copying pycocotools\cocoeval.py -> build\lib.win-amd64-3.7\pycocotools
      copying pycocotools\mask.py -> build\lib.win-amd64-3.7\pycocotools
      copying pycocotools\__init__.py -> build\lib.win-amd64-3.7\pycocotools
      running build_ext
      cythoning pycocotools/_mask.pyx to pycocotools\_mask.c
      c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\Cython\Compiler\Main.py:369: FutureWarning: Cython directive 'language_level' not set, using 2 for now (Py2). This will change in a later release! File: C:\Users\APIDWA~1\AppData\Local\Temp\pip-install-jt0cdwch\pycocotools_ad600866e39342aa97877c453bcad251\pycocotools\_mask.pyx
        tree = Parsing.p_module(s, pxd, full_module_name)
      building 'pycocotools._mask' extension
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      ----------------------------------------
      ERROR: Failed building wheel for pycocotools
        ERROR: Command errored out with exit status 1:
         command: 'c:\users\apidwalin\anaconda3\envs\tf\python.exe' -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\APIDWA~1\\AppData\\Local\\Temp\\pip-install-jt0cdwch\\pycocotools_ad600866e39342aa97877c453bcad251\\setup.py'"'"'; __file__='"'"'C:\\Users\\APIDWA~1\\AppData\\Local\\Temp\\pip-install-jt0cdwch\\pycocotools_ad600866e39342aa97877c453bcad251\\setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record 'C:\Users\APIDWA~1\AppData\Local\Temp\pip-record-t45xcqg8\install-record.txt' --single-version-externally-managed --compile --install-headers 'c:\users\apidwalin\anaconda3\envs\tf\Include\pycocotools'
             cwd: C:\Users\APIDWA~1\AppData\Local\Temp\pip-install-jt0cdwch\pycocotools_ad600866e39342aa97877c453bcad251\
        Complete output (14 lines):
        running install
        running build
        running build_py
        creating build
        creating build\lib.win-amd64-3.7
        creating build\lib.win-amd64-3.7\pycocotools
        copying pycocotools\coco.py -> build\lib.win-amd64-3.7\pycocotools
        copying pycocotools\cocoeval.py -> build\lib.win-amd64-3.7\pycocotools
        copying pycocotools\mask.py -> build\lib.win-amd64-3.7\pycocotools
        copying pycocotools\__init__.py -> build\lib.win-amd64-3.7\pycocotools
        running build_ext
        skipping 'pycocotools\_mask.c' Cython extension (up-to-date)
        building 'pycocotools._mask' extension
        error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        ----------------------------------------
    ERROR: Command errored out with exit status 1: 'c:\users\apidwalin\anaconda3\envs\tf\python.exe' -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\APIDWA~1\\AppData\\Local\\Temp\\pip-install-jt0cdwch\\pycocotools_ad600866e39342aa97877c453bcad251\\setup.py'"'"'; __file__='"'"'C:\\Users\\APIDWA~1\\AppData\\Local\\Temp\\pip-install-jt0cdwch\\pycocotools_ad600866e39342aa97877c453bcad251\\setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record 'C:\Users\APIDWA~1\AppData\Local\Temp\pip-record-t45xcqg8\install-record.txt' --single-version-externally-managed --compile --install-headers 'c:\users\apidwalin\anaconda3\envs\tf\Include\pycocotools' Check the logs for full command output.
    

Get `tensorflow/models` or `cd` to parent directory of the repository.


```python
import os
import pathlib


if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models
```

Compile protobufs and install the object_detection package


```bash
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
```

    Couldn't find program: 'bash'
    


```bash
%%bash 
cd models/research
pip install .
```

    Couldn't find program: 'bash'
    

### Imports


```python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
```

Import the object detection module.


```python
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
```

Patches:


```python
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile
```

# Model preparation 

## Variables

Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.

By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

## Loader


```python
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model
```

## Loading label map
Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


```python
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
```

For the sake of simplicity we will test on 2 images:


```python
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS
```




    [WindowsPath('models/research/object_detection/test_images/dog&cat.jpg'),
     WindowsPath('models/research/object_detection/test_images/image1.jpg'),
     WindowsPath('models/research/object_detection/test_images/image2.jpg'),
     WindowsPath('models/research/object_detection/test_images/puppy.jpg')]



# Detection

Load an object detection model:


```python
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)
```

    WARNING:tensorflow:From C:\Users\APIDWA~1\AppData\Local\Temp/ipykernel_2120/1723068443.py:11: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    C:\Users\APIDWA~1\AppData\Local\Temp/ipykernel_2120/2266027781.py in <module>
          1 model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    ----> 2 detection_model = load_model(model_name)
    

    C:\Users\APIDWA~1\AppData\Local\Temp/ipykernel_2120/1723068443.py in load_model(model_name)
          9   model_dir = pathlib.Path(model_dir)/"saved_model"
         10 
    ---> 11   model = tf.saved_model.load(str(model_dir))
         12 
         13   return model
    

    c:\users\apidwalin\anaconda3\envs\tf\lib\site-packages\tensorflow_core\python\util\deprecation.py in new_func(*args, **kwargs)
        322               'in a future version' if date is None else ('after %s' % date),
        323               instructions)
    --> 324       return func(*args, **kwargs)
        325     return tf_decorator.make_decorator(
        326         func, new_func, 'deprecated',
    

    TypeError: load() missing 2 required positional arguments: 'tags' and 'export_dir'


Check the model's input signature, it expects a batch of 3-color images of type uint8:


```python
print(detection_model.signatures['serving_default'].inputs)
```

And returns several outputs:


```python
detection_model.signatures['serving_default'].output_dtypes
```


```python
detection_model.signatures['serving_default'].output_shapes
```

Add a wrapper function to call the model, and cleanup the outputs:


```python
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict
```

Run it on each test image and show the results:


```python
def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  display(Image.fromarray(image_np))
```


```python
for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)

```

## Instance Segmentation


```python
model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
masking_model = load_model(model_name)
```

The instance segmentation model includes a `detection_masks` output:


```python
masking_model.output_shapes
```


```python
for image_path in TEST_IMAGE_PATHS:
  show_inference(masking_model, image_path)
```
