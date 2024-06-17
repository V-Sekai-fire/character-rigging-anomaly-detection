> Assume that we have a dataset in which the training set contains only normal images, and the test set contains both normal and abnormal images. We also have masks for the abnormal images in the test set. We want to train an anomaly segmentation model that will be able to detect the abnormal regions in the test set.
> 
> There are certain cases where we only have normal images in our dataset but would like to train a segmentation model. This could be done in two ways:
Use the synthetic anomaly generation feature to create abnormal images from normal images, and perform the validation and test steps.

[]

```
pip3 install anomalib
anomalib install --option core
```

In vr generate photos of good images.
