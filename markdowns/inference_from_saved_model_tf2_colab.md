# Intro to Object Detection Colab

Welcome to the object detection colab! This demo will take you through the steps of running an "out-of-the-box" detection model in SavedModel format on a collection of images.



Imports


```python
!pip install -U --pre tensorflow=="2.2.0"
```

    Collecting tensorflow==2.2.0
      Downloading tensorflow-2.2.0-cp37-cp37m-win_amd64.whl (459.2 MB)
    

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
    


```python
import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models
```


```python
# Install the Object Detection API
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```


      File "C:\Users\APIDWA~1\AppData\Local\Temp/ipykernel_10400/820332112.py", line 3
        cd models/research/
                ^
    SyntaxError: invalid syntax
    



```python
import io
import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

%matplotlib inline
```


```python
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Load the COCO Label Map
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    67: {'id': 67, 'name': 'dining table'},
    70: {'id': 70, 'name': 'toilet'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'},
}
```


```python
# Download the saved model and put it into models/research/object_detection/test_data/
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz
!tar -xf efficientdet_d5_coco17_tpu-32.tar.gz
!mv efficientdet_d5_coco17_tpu-32/ models/research/object_detection/test_data/
```


```python
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('models/research/object_detection/test_data/efficientdet_d5_coco17_tpu-32/saved_model/')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')
```


```python
import time

image_dir = 'models/research/object_detection/test_images'

elapsed = []
for i in range(2):
  image_path = os.path.join(image_dir, 'image' + str(i + 1) + '.jpg')
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = np.expand_dims(image_np, 0)
  start_time = time.time()
  detections = detect_fn(input_tensor)
  end_time = time.time()
  elapsed.append(end_time - start_time)

  plt.rcParams['figure.figsize'] = [42, 21]
  label_id_offset = 1
  image_np_with_detections = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)
  plt.subplot(2, 1, i+1)
  plt.imshow(image_np_with_detections)

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
```
