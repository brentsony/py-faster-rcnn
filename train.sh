# Train a Faster R-CNN model:
# INPUT:   data/VOCdevkit2007/VOC2007/   (must be in VOC format)
# OUTPUT:  
# ln -s faster_rcnn_alt_opt/voc_2007_trainval/VGG_CNN_M_1024_faster_rcnn_final.caffemodel  data/faster_rcnn_models/SONYNET_faster_rcnn_final.caffemodel

#MODEL=VGG_CNN_M_1024
MODEL=VGG16

rm -rf data/cache
experiments/scripts/faster_rcnn_alt_opt.sh 0 $MODEL pascal_voc
