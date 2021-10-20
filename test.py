import cv2
from TF_OD import predict


for sample in ['sample_images/Cars_01',
               'sample_images/Cars_02',
               'sample_images/Cars_03',
               'sample_images/Cars_04']:

    img, box = predict(f'{sample}.png')
    cv2.imwrite(f'{sample}_number_plate.png', img)
    # cv2.imshow(f'{sample}_number_plate', img[box[0]:box[1], box[2]:box[3]])
    # cv2.waitKey(0)
