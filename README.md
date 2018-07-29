# deeplab_imseg
customize Google's DeepLab IMSEG with my appetite 

### git clone this repo
> cd ~ && git clone https://github.com/syyunn/deeplab_imseg

### put images you want to seg into the "img" folder 

### run the segmentation 
> cd ~/deeplab_imseg && python run.py

#it takes a while at the first to download 4 different tf models
#each model name is one of ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval','xception_coco_voctrainaug', 'xception_coco_voctrainval'] with the size of 24MB, 24MB, 420MB, 420MB resp.

### these names could be used as a option of run.py such as
> python run.py --model mobilenetv2_coco_voctrainaug
