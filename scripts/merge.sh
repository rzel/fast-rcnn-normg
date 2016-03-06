PYTHONPATH='.' python python_utils/do_net_surgery.py \
  --out_net_def scripts/joint_rgbnorm/test.prototxt.images+hha \
  --net_surgery_json scripts/joint_rgbnorm/init_weights.json \
  --out_net_file /nfs.yoda/xiaolonw/fast_rcnn/models_norm/alexnet_rgb/fast_rcnn_joint.caffemodel

