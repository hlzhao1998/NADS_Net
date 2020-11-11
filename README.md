# NADS-Net: A Nimble Architecture for Driver and Seat Belt Detection via Convolutional Neural Networks



This is an **approximate** implementation of [NADS-Net: A Nimble Architecture for Driver and Seat Belt Detection via Convolutional Neural Networks](https://arxiv.org/abs/1910.03695) with Pytorch. 

The code of NADS-Net is not public. Considering that the work of NADS-Net is based on [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008) , we **approximately** implement the NADS-Net based on an implementation of OpenPose:[an implementation of PyTorch OpenPose](https://github.com/tracer9/Pytorch0.4.1_Openpose).



## Attension!

NADS-Net is a network  with there detection head, i.e. the keypoints detection head(outputs Confidence Maps), the PAF detection head, and a seat belt segmentation head. Our application scenario is to estimate the pose of the drivers to detect and prevent possible dangerous driving behaviors. Status of seat belts matters a lot in the scenario. However, there is **EXTREMELY FEW** public datasets with seat belt labeling. Considering that, the influence of safety belt on driving pose estimation is not included in our application, i.e. we **DID NOT** implement the seat belt detection head! We only detect the pose of the driver(by only the keypoints detection head and the PAF detection head) to assess driving safety.



