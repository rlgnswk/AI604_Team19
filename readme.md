# ZSRGAN: Zero-shot Super-Resolution with Generative Adversarial Network(Pytorch)
### 2021-fall-AI604-KAIST /  TEAM 19

[Report](https://github.com/rlgnswk/ZSRGAN/blob/main/Report/ZSRGAN_Final_Report.pdf)

[Presentation](https://www.youtube.com/watch?v=EquVMzSdkHo&list=PLgaQUWOjONyV9f1LK30reH6gCj14PcUot)

-----------------

# Abstract:
Deep learning based super-resolution (SR) is one of the
most actively studied areas of computer vision. However,
many of these studies are conducted on a supervised manner, requiring a large amount of data. There are several
problems with this. First, the ground truth and input pair of
the dataset is made using only a specific procedure, usually
the bicubic downsampling. As a result, supervised SR works
well only for these images, introducing a second problem
where the model can not work well for test images not found
in the training distribution. Based on this, it is difficult to
say that these methods indeed is super-resolution for raw
images found in the real world. In this paper, we introduce a
novel network named Zero-shot Super-Resolution with Generative Adversarial Network (ZSRGAN) for real world image SR, which needs only one test image and does not rely
on any other external datasets. Unlike existing methods,
we propose optimization on the perceptual aspect as well
as reconstruction of pixel units using zero-shot SR method.
Therefore, through our proposed model, a real world image can be super resolved with the best perceptual quality
without any information from additional datasets.


## Pipeline

## ![pipline](./figs/pipline.png)

## Results:

![Result1](./figs/Result1.png)



![Result2](./figs/Result2.png)

----------
# Usage:

## Run on sample data:
First, the sample data(Degraded Set5) already are placed in ```<ZSRGAN_path>/datasets/MySet5```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path>
```
## Run on your data:
You can find additional dataset 
from [Here](https://drive.google.com/file/d/16L961dGynkraoawKE2XyiCh4pdRS-e4Y/view) 
provided by [MZSR](https://github.com/JWSoh/MZSR) (CVPR 2020)

First, put your data files in ```<ZSRGAN_path>/datasets/```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path> --dataset <name_of_your_dataset> --GT_path <HR_folder_in_your_dataset> --LR_path <LR_folder_in_your_dataset>
```
# References
Our project was based on [ZSSR](https://github.com/assafshocher/ZSSR) (CVPR 2018) and the data was taken from [MZSR](https://github.com/JWSoh/MZSR) (CVPR 2020).
