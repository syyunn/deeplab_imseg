import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import numpy as np
from PIL import Image
import tensorflow as tf
import argparse
import glob

from src.DeepLabModel import DeepLabModel, label_to_color_image

parser = argparse.ArgumentParser()

parser.add_argument("--model", default= "xception_coco_voctrainval", choices=['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval','xception_coco_voctrainaug', 'xception_coco_voctrainval'])

args, unknown = parser.parse_known_args()

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def vis_segmentation(image, seg_map):
    import numpy as np
    """Visualizes input image, segmentation map and overlay view."""
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    return seg_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    check = ['./models/mobilenetv2_coco_voctrainaug.tar.gz',
 './models/xception_coco_voctrainval.tar.gz',
 './models/xception_coco_voctrainaug.tar.gz',
 './models/mobilenetv2_coco_voctrainval.tar.gz']
    download = []
    for i in range(len(check)):
        if not os.path.isfile(check[i]):
            download.append(check[i].split('/')[2].split('.')[0])
    
    #@title Select and download models {display-mode: "form"}
    MODEL_CAND = download

    for i in range(len(MODEL_CAND)):
        MODEL_NAME = MODEL_CAND[i]

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        _TARBALL_NAME = MODEL_NAME+'.tar.gz'

        model_dir = "./models/"
        tf.gfile.MakeDirs(model_dir)

        download_path = os.path.join(model_dir, _TARBALL_NAME)
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                           download_path)
        print('download completed! loading DeepLab model...')
    
    path = "./models/" + args.model + ".tar.gz"
    
    MODEL = DeepLabModel(path)
    print('model loaded successfully!')
    
    
    images = []
    im_type= ["png", "jpg", "jpeg"]
    for i in range(len(im_type)):
        print(glob.glob("./img/*."+im_type[i]))
        images = images + glob.glob("./img/*."+im_type[i])
        
    for i in range(len(images)):
        with open(images[i], "rb") as f:
            jpeg_str = f.read()
        original_im = Image.open(BytesIO(jpeg_str))
        resized_im, seg_map = MODEL.run(original_im)
        seg_image = Image.fromarray(vis_segmentation(resized_im, seg_map))
        seg_image.save("./out/"+images[i].split('/')[2])

