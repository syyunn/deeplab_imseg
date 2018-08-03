class DeepLabModel(object): #object gets filepath of inference model
  """Class to load deeplab model and run inference."""
  
  
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    import tensorflow as tf
    import tarfile
    import os

    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    self.sess = tf.Session(graph=self.graph, config=config)

  def run(self, image):
    from PIL import Image
    import numpy as np

    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    #width, height = image.size
    #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (256, 256)
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map #this will be our result we want 


def create_pascal_label_colormap():
  import numpy as np
  """ Creates a label colormap used in PASCAL VOC segmentation benchmark.
  
  Returns:
    A Colormap for visualizing segmentation results.
  """

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
    import numpy as np
    """Adds color defined by the dataset colormap to the label.
        
        Args:
        label: A 2D array with integer type, storing the segmentation label.
        
        Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
        
        Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
        """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
                    
    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
                            
    return colormap[label]

