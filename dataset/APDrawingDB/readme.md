# APDrawing Dataset

Our APDrawing Dataset contains 140 pairs of face photos and corresponding artistic portrait drawings.
All portrait drawings were drawn by one single professional artist.

All images and drawings are aligned, downsampled and cropped to 512x512 size for training and testing.
70 image pairs as the training set and the remaining 70 image pairs as the test set, as shown in `data` folder (the `train` subfolder contains augmented data).

This dataset folder is organized as follows:
```
    /data -- the aligned data
        /train -- aligned train images (augmented)
        /test -- aligned test images
    /landmark -- 5 facial landmarks of train and test images
        /ALL -- contains all landmark files
    /mask -- background masks of train and test images
        /ALL -- contains all background masks
``` 

Note:
- 1.Five facial landmarks in folder *landmark/* are detected using MTCNN in paper:
Zhang, K., Zhang, Z., Li, Z., & Yu, Q. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), 1499-1503.
Using code in https://github.com/kpzhang93/MTCNN_face_detection_alignment (MTCNNv1).
- 2.Background mask in folder *mask/* is segmented by method in
"Automatic Portrait Segmentation for Image Stylization"
Xiaoyong Shen, Aaron Hertzmann, Jiaya Jia, Sylvain Paris, Brian Price, Eli Shechtman, Ian Sachs. Computer Graphics Forum, 35(2)(Proc. Eurographics), 2016.
Using code in http://xiaoyongshen.me/webpage_portrait/index.html

## Citation
If you use these data, please cite our paper:
```
@inproceedings{YiLLR19,
  title     = {{APDrawingGAN}: Generating Artistic Portrait Drawings from Face Photos with Hierarchical GANs},
  author    = {Yi, Ran and Liu, Yong-Jin and Lai, Yu-Kun and Rosin, Paul L},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR '19)},
  pages     = {10743--10752},
  year      = {2019}
}
```
