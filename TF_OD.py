import os
import tensorflow as tf

import cv2
import numpy as np
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = './Tensorflow/workspace/models/my_ssd_mobnet'
files= {
    'PIPELINE_CONFIG' : f'{path}/pipeline.config',
    'LABELMAP' : './Tensorflow/workspace/annotations/label_map.pbtxt'
    }

paths = {'CHECKPOINT_PATH' : path}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-12')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def predict(img_path):

    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

    img = cv2.imread(img_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    ymin, xmin, ymax, xmax = detections['detection_boxes'][0]

    height, width, _ = image_np.shape
    y1 = int(ymin * height)
    x1 = int(xmin * width)
    y2 = int(ymax * height)
    x2 = int(xmax * width)

    # print(x1, y1, x2, y2)

    min_score_thresh = detections['detection_scores'][1]

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=min_score_thresh,
                agnostic_mode=False)

    return image_np_with_detections, (y1, y2, x1, x2)

    # cv2.imwrite('result.jpg', cv2.cvtColor(image_np_with_detections[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # plt.show()
