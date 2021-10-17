#!/bin/sh -x

BASE_MODEL_PATH='/lhome/zhangwm/git/ai-bench/net/'
MODEL_PATH='/lhome/zhangwm/git/ai-bench/net/caffe/'
TF_MODEL_PATH='/lhome/zhangwm/git/ai-bench/net/tensorflow/'
LAYER_MODEL_PATH='/home/zhangwm/tmp/pb/'
IMAGE_PATH='/lhome/zhangwm/git/ai-bench/config_file/'

#python3 hhb.py -f ${MODEL_PATH}/mobilenet/mobilenetv1.prototxt ${MODEL_PATH}/mobilenet/mobilenetv1.caffemodel --calibrate-dataset ${IMAGE_PATH}/cat.jpg --data-scale 0.007843 --data-mean '104 117 124' --board gref -C
#python3 hhb.py -f ${MODEL_PATH}/mobilenet/mobilenetv1.prototxt ${MODEL_PATH}/mobilenet/mobilenetv1.caffemodel --calibrate-dataset ${IMAGE_PATH}/cat.jpg --data-scale 0.007843 --data-mean '104 117 124' --board anole -C
python3 hhb.py -f ${MODEL_PATH}/mobilenet/mobilenetv1.prototxt ${MODEL_PATH}/mobilenet/mobilenetv1.caffemodel --calibrate-dataset ${IMAGE_PATH}/cat.jpg --data-scale 0.017 --data-mean '104 117 124' --board x86_ref --simulate -sd ${IMAGE_PATH}/cat.jpg
#python3 hhb.py -f ${MODEL_PATH}/resnet/resnet50.prototxt ${MODEL_PATH}/resnet/resnet50.caffemodel -cd ${IMAGE_PATH}/cat.jpg --data-scale 1 --data-mean "0 0 0" --board light -C
#python3 hhb.py -f ${MODEL_PATH}/resnet/resnet50.prototxt ${MODEL_PATH}/resnet/resnet50.caffemodel -cd ${IMAGE_PATH}/cat.jpg --data-scale 1 --data-mean "0 0 0" --board x86_ref --simulate -sd ${IMAGE_PATH}/cat.jpg
#python3 hhb.py -f ${MODEL_PATH}/inception/inceptionv3.prototxt ${MODEL_PATH}/inception/inceptionv3.caffemodel -cd ${IMAGE_PATH}/cat.jpg -s 0.007843 -m "104 117 124" --board anole -C

#python3 hhb.py -C -f ${TF_MODEL_PATH}/inception/inceptionv1.pb -is "1 3 224 224" -in "input" -on "InceptionV1/Logits/Predictions/Reshape_1" -cd ${IMAGE_PATH}/cat.jpg
#python3 hhb.py -C -f ${TF_MODEL_PATH}/densenet/densenet121.pb -is "1 3 224 224" -in "Placeholder" -on "densenet121/predictions/Reshape_1" -cd ${IMAGE_PATH}/cat.jpg

#python3 hhb.py --caffe-proto ${MODEL_PATH}/lenet/lenet.prototxt --caffe-blobs ${MODEL_PATH}/lenet/lenet.caffemodel --dataset ${IMAGE_PATH}/1.jpg --input-mean 63.5 --input-normal 63.5
#python3 hhb.py --caffe-proto ${MODEL_PATH}/lenet/lenet.prototxt --caffe-blobs ${MODEL_PATH}/lenet/lenet.caffemodel --dataset ${IMAGE_PATH}/1.jpg --input-mean 63.5 --input-normal 63.5 --simulate --postprocess top5
#python3 hhb.py --caffe-proto ${MODEL_PATH}/mobilenet/mobilenetv1.prototxt --caffe-blobs ${MODEL_PATH}/mobilenet/mobilenetv1.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 127.5 --input-mean "104 117 124"
#python3 hhb.py --caffe-proto ${MODEL_PATH}/mobilenet/mobilenetv1.prototxt --caffe-blobs ${MODEL_PATH}/mobilenet/mobilenetv1.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 127.5 --input-mean "104 117 124" --simulate --postprocess top5 --no-quantize
#python3 hhb.py --caffe-proto ${MODEL_PATH}/mobilenet/mobilenetv2.prototxt --caffe-blobs ${MODEL_PATH}/mobilenet/mobilenetv2.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 128.5 --input-mean "104 117 124"
#python3 hhb.py --caffe-proto ${MODEL_PATH}/inception/inceptionv1.prototxt --caffe-blobs ${MODEL_PATH}/inception/inceptionv1.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 1 --input-mean "0 0 0" --simulate --postprocess top5
#python3 hhb.py --caffe-proto ${MODEL_PATH}/inception/inceptionv3.prototxt --caffe-blobs ${MODEL_PATH}/inception/inceptionv3.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 127.5 --input-mean "104 117 124"
#python3 hhb.py --caffe-proto ${MODEL_PATH}/inception/inceptionv4.prototxt --caffe-blobs ${MODEL_PATH}/inception/inceptionv4.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 127.5 --input-mean "104 117 124"
#python3 hhb.py --caffe-proto ${MODEL_PATH}/resnet/resnet50.prototxt --caffe-blobs ${MODEL_PATH}/resnet/resnet50.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 1 --input-mean "0 0 0"
#python3 hhb.py --caffe-proto ${MODEL_PATH}/resnet/resnet50.prototxt --caffe-blobs ${MODEL_PATH}/resnet/resnet50.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 1 --input-mean "0 0 0" --simulate --postprocess top5
#python3 hhb.py --caffe-proto ${MODEL_PATH}/vgg/vgg16.prototxt --caffe-blobs ${MODEL_PATH}/vgg/vgg16.caffemodel --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --caffe-proto ${MODEL_PATH}/ssd/ssdmobilenetv1.prototxt --caffe-blobs ${MODEL_PATH}/ssd/ssdmobilenetv1.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --simulate --postprocess top5
#python3 hhb.py --caffe-proto ${MODEL_PATH}/ssd/ssdmobilenetv2.prototxt --caffe-blobs ${MODEL_PATH}/ssd/ssdmobilenetv2.caffemodel --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --caffe-proto ${MODEL_PATH}/squeezenet/squeezenet_v1.0.prototxt --caffe-blobs ${MODEL_PATH}/squeezenet/squeezenet_v1.0.caffemodel --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --caffe-proto ${MODEL_PATH}/enet/enet.prototxt --caffe-blobs ${MODEL_PATH}/enet/enet.caffemodel --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --caffe-proto ${BASE_MODEL_PATH}/anole/citybrain/yolov3/yolov3_citybrain.prototxt --caffe-blobs ${BASE_MODEL_PATH}/anole/citybrain/yolov3/yolov3_citybrain.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-shape "1 3 416 416" --input-name "data"
#python3 hhb.py --dataset ${IMAGE_PATH}/rfcn.npz --caffe-proto ${MODEL_PATH}/rfcn/resnet101_rfcn_voc.prototxt --caffe-blobs ${MODEL_PATH}/rfcn/resnet101_rfcn_voc.caffemodel --input-mean "103.94 116.98 123.68" --input-normal 127.5
#python3 hhb.py --dataset /home/zhangwm/git/ai-bench/net/anole/citybrain/ssd/0001.jpg --caffe-proto /home/zhangwm/git/ai-bench/net/anole/citybrain/ssd/deploy.prototxt --caffe-blobs /home/zhangwm/git/ai-bench/net/anole/citybrain/ssd/ssd-unet.caffemodel --input-mean "103.94 116.98 123.68" --input-normal 127.5 --simulate --postprocess top5


#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/inception/inceptionv1.pb --input-shape "1 3 224 224" --input-name "input" --output-name "InceptionV1/Logits/Predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/inception/inceptionv3.pb --input-shape "1 3 299 299" --input-name "input" --output-name "InceptionV3/Predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/inception/inceptionv4.pb --input-shape "1 3 299 299" --input-name "input" --output-name "InceptionV4/Logits/Predictions" --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/resnet/resnet50.pb --input-shape "1 3 224 224" --input-name "input" --output-name "resnet_v1_50/predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/vgg/vgg16.pb --input-shape "1 3 224 224" --input-name "input" --output-name "vgg_16/fc8/squeezed" --dataset ${IMAGE_PATH}/cat.jpg
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/mobilenet/mobilenet_v1_1.0_224.pb --input-shape "1 3 224 224" --input-name "input" --output-name "MobilenetV1/Predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg


#python3 hhb.py --caffe-proto ${MODEL_PATH}/resnet/resnet50.prototxt --caffe-blobs ${MODEL_PATH}/resnet/resnet50.caffemodel --dataset ${IMAGE_PATH}/cat.jpg --input-normal 1 --input-mean "0 0 0" --board c860

#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/alexnet/alexnet.pb --input-shape "1 3 227 227" --input-name "Placeholder" --output-name "Softmax" --dataset ${IMAGE_PATH}/cat.jpg --simulate --postprocess top5 #--board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/alexnet/alexnet.pb --input-shape "1 3 227 227" --input-name "Placeholder" --output-name "Softmax" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/mobilenet/mobilenet_v1_1.0_224.pb --input-shape "1 3 224 224" --input-name "input" --output-name "MobilenetV1/Predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/inception/inceptionv1.pb --input-shape "1 3 224 224" --input-name "input" --output-name "InceptionV1/Logits/Predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/inception/inceptionv3.pb --input-shape "1 3 299 299" --input-name "input" --output-name "InceptionV3/Predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/inception/inceptionv4.pb --input-shape "1 3 299 299" --input-name "input" --output-name "InceptionV4/Logits/Predictions" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/resnet/resnet50.pb --input-shape "1 3 224 224" --input-name "input" --output-name "resnet_v1_50/predictions/Reshape_1" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/vgg/vgg16.pb --input-shape "1 3 224 224" --input-name "input" --output-name "vgg_16/fc8/squeezed" --dataset ${IMAGE_PATH}/cat.jpg --board c860
#python3 hhb.py --tf-pb ${TF_MODEL_PATH}/yolo/yolov3.pb --input-shape "1 3 416 416" --input-name "Placeholder" --output-name "conv_sbbox/BiasAdd;conv_mbbox/BiasAdd;conv_lbbox/BiasAdd" --dataset ${IMAGE_PATH}/cat.jpg --board c860

#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/sin/sin_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "sin" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/asin/asin_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "asin" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/asinh/asinh_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "asinh" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/sinh/sinh_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "sinh" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/cos/cos_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "cos" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/acos/acos_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "acos" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/cosh/cosh_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "cosh" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/acosh/acosh_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "acosh" --dataset /mnt/tmp/pb/input1.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/tan/tan_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "tan" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/atan/atan_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "atan" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/tanh/tanh_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "tanh" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/atanh/atanh_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "atanh" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/segment_max/segment_max_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "segment_max" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/segment_min/segment_min_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "segment_min" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/segment_mean/segment_mean_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "segment_mean" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/segment_prod/segment_prod_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "segment_prod" --dataset /mnt/tmp/pb/input.npz --board c860
#python3 hhb.py --tf-pb ${LAYER_MODEL_PATH}/segment_sum/segment_sum_model.pb --input-shape "3 224 224" --input-name "Placeholder" --output-name "segment_sum" --dataset /home/zhangwm/tmp/pb/input.npz --board c860
