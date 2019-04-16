# RepNet-Vehicle-ReID
Vehicle re-identification implementing RepNet

## Vehicle ReID task: </br>
![](https://github.com/CaptainEven/RepNet-Vehicle-ReID/blob/master/VehicleReIDTask.png)

## Basic principle for vehicle ReID task: </br>
Using a two-branch deep convolutional network to project raw vehicle images into an Euclidean space where distance can be directly used to measure the similarity of arbitrary two vehicles. </r>
For simplicity, triplet loss or coupled cluster loss is replaced here by arc loss which is widely used in face recognition.

# Test result
![](https://github.com/CaptainEven/RepNet-Vehicle-ReID/blob/master/TestResult.png)

## Network structure: </br>
![](https://github.com/CaptainEven/RepNet-Vehicle-ReID/blob/master/RepNet.png)
![](https://github.com/CaptainEven/RepNet-Vehicle-ReID/blob/master/RepNet2.png)

## Reference: </br>
[Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Relative_Distance_CVPR_2016_paper.pdf) </br>
[Learning a repression network for precise vehicle search](https://arxiv.org/pdf/1708.02386.pdf) </br>

## Dataset: </br>
[VehicleID dataset](https://pan.baidu.com/s/1JKOysKjrlgReuxZ2ONCmUQ) </br>

## Pre-trained model
[model](https://pan.baidu.com/s/1vJiwBfR3f9Zc9NCuUbmEsw) </br>
extract code: 62wn
