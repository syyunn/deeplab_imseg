# deeplab_imseg
Customized Google's DeepLab IMSEG to use local img paths 

### Git clone this repo
> cd ~ && git clone https://github.com/syyunn/deeplab_imseg

### Put images you want to seg into the "img" folder 
> Just use GUI dude

### Run the segmentation 
> cd ~/deeplab_imseg && python run.py

#it takes a while at the first to download 4 different tf models
#each model name is one of ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval','xception_coco_voctrainaug', 'xception_coco_voctrainval'] with the size of 24MB, 24MB, 420MB, 420MB resp.

### Those model names could be used as an option of run.py 
> python run.py --model mobilenetv2_coco_voctrainaug
#use small modles if your task is simple
