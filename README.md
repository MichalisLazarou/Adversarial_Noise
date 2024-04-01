# Adversarial Noise exercise

## Problem
The problem is to take as input an image and a target class and output the same image with added adversarial noise until it is classified in the target calss.

## Analysis.

First of all I tried to understand the problem of Adversarial attacks by reading the available resources and the problem exercise. Then I decided to use Python and the Pytorch framework to implement my solution because it was suggested from the exercise. I used a pre-trained image classification model from torchvision, specifically the efficientnet_v2_m because it has very good accuracy performance and its size is reasonable. The procedure is the same with any other model I would have chosen from torchvision, therefore I did not spend too much time in trying to analyze which pre-trained model to use. Then I looked for available resources to implement adversarial attacks and I found torchattacks library which seemed to be very helpful since it had implemented a lot of the state of the art methods. 

I analyzed a few of the available adversarial methods and found out that the methods proposed in "‘Towards Deep Learning Models Resistant to Adversarial Attacks" and "EAD: Elastic-Net Attacks to Deep Neural Networks" seem to provide the best results for this exercise. I decided to support both methods in my code, with the default being the PGD and in case the users wants he can pass a parameter to use the EAD method. I verified these methods by testing them on available dataset from ImageNet as shown in `verification_procedure.py`. I used 500 images and calculated the accuracy (whether the image can be classified in the wrong class with as little adversarial noise as limited). Note that I kept adding noise until the image is classified to the target class which means that in certain occasions too much noise may be added. Additionally, in order to choose which method to use I visualized 50 images by saving them and decided which method satisfies the requirements of the exercise the best.

Thinking that this would potentially be a small library, I decided to make it into a Class with a few simple functions, one that is used in the `main.py` at the inference stage where adversarial attack is carried out when the path of an image is provided only, another that is was used in the verification procedure and works by getting as input images as tensors and another that is used to save the image.

## Limitations

There are some limitations in this method. One is that the noise may be perceptible in certain occasion. This is because the classifier was classifies an image over 1000 classes (Imagenet classifier) and in order to make an image to be classified in a specific class a lot of noise may be required to be added. A way to bypass this problem would have been simply to use a smaller set of classes i.e. 10. This would make the model require much less noise to classify an image to a target class since the set of potential classes is much smaller. However, this does not really solve the problem but simply simplifies it and I decided to go for the more realistic one with 1000 classes. Also, since it was suggested to use a pre-trained model from torchvision and most models are trained on ImageNet I felt that was the best thing to do.

## Future work.

In the future, I am interested in investigating more recent state of the art methods perhaps ones that have been published recently in top conferences such as CVPR, ICCV, ECVV, ICML, NeurIPS and attempt to use them in this problem. Additionally, I am interested in using image perception scores such as LPIPS which stands for the perceptual similarity between two images. LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.

## Environment configuration

- Python 3.9
```bash
pip install -r requirements.txt
```

## Run example

```bash
python main.py --target_class <CLASS_NAME> --img_path <PATH_OF_IMAGE>
```
In case you are interested in using the EAD attack please run

```bash
python main.py --target_class <CLASS_NAME> --img_path <PATH_OF_IMAGE> --attack EAD
```