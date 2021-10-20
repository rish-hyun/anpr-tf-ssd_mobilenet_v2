# anpr-tf-ssd_mobilenet_v2
[![](https://img.shields.io/badge/Python-Tensorflow-blue)](https://pypi.org/project/tensorflow/)

Automatic number-plate recognition built using [tf-ssd_mobilenet_v2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)
<hr>

* The [tf-ssd_mobilenet_v2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2) model is trained using transfer learning (fine-tuned).
* The annotated dataset used for learning is taken from [kaggle](https://www.kaggle.com/andrewmvd/car-plate-detection).
* Trained steps `--num_train_steps=10000`

Trained model is evaluated in [TF_OD.py](https://github.com/rish-hyun/anpr-tf-ssd_mobilenet_v2/blob/main/TF_OD.py)

```python
import cv2
from TF_OD import predict


for sample in ['sample_images/Cars_01',
               'sample_images/Cars_02',
               'sample_images/Cars_03']:

    img, box = predict(f'{sample}.png')
    cv2.imshow(sample, img)
    # cv2.imshow(f'{sample}_number_plate', img[box[0]:box[1], box[2]:box[3]])
    cv2.waitKey(0)

```

<hr>

### Results
<p>
<img src="https://github.com/rish-hyun/anpr-tf-ssd_mobilenet_v2/blob/main/sample_images/Cars_01_number_plate.png" width="250" height="166" />
<img src="https://github.com/rish-hyun/anpr-tf-ssd_mobilenet_v2/blob/main/sample_images/Cars_02_number_plate.png" width="250" height="166" />
<img src="https://github.com/rish-hyun/anpr-tf-ssd_mobilenet_v2/blob/main/sample_images/Cars_03_number_plate.png" width="250" height="166" />
</p>

<hr>

In the next step, OCR can be applied to ROI (number-plate detected region)

* For this, any model can be used such as such as _KerasOCR, PyTesseract, EasyOCR_, etc.
* I trained a simple [OCR](https://github.com/rish-hyun/anpr-tf-ssd_mobilenet_v2/blob/main/ocr_model.h5) model from scratch using _**MNIST**_ and _**Kaggle**_ datasets.

The code for training OCR can be found in [OCR.ipynb](https://github.com/rish-hyun/anpr-tf-ssd_mobilenet_v2/blob/main/OCR.ipynb)

### Final Note
OCR Model is not at its best. A trained _Attention OCR Model_ would perform a lot better.
