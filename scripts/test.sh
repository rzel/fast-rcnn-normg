# Testing alexnet_rgb and alexnet_hha models
model='alexnet_rgb_alexnet_hha'; tr_set='trainval'; test_set='test'; modality="images+norm";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def scripts/joint_rgbnorm/test.prototxt.images+hha \
  --net /nfs.yoda/xiaolonw/fast_rcnn/models_norm/alexnet_rgb/fast_rcnn_joint.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg scripts/joint_rgbnorm/config.prototxt


