# Sign-Language-Digits-Dataset
  Dataset used : https://github.com/ardamavi/Sign-Language-Digits-Dataset
  
    Image size: 100x100
  
    Color space: RGB
  
### Sample Images:

![example_0](https://user-images.githubusercontent.com/19996897/37919353-443e3636-3141-11e8-8f3e-d9c5e18060cf.JPG) 
0
![example_1](https://user-images.githubusercontent.com/19996897/37919356-44afb1ee-3141-11e8-8876-0a2483d82dc0.JPG) 
1
![example_2](https://user-images.githubusercontent.com/19996897/37919357-44fa6fe0-3141-11e8-8ac2-3cb1c47c17a6.JPG) 
2
![example_3](https://user-images.githubusercontent.com/19996897/37919358-4543f35e-3141-11e8-8736-0e8624221301.JPG) 
3
![example_4](https://user-images.githubusercontent.com/19996897/37919359-45970986-3141-11e8-86ed-c7a73aab6ae8.JPG) 
4
![example_5](https://user-images.githubusercontent.com/19996897/37919360-45f79c06-3141-11e8-9f7e-81900b73baab.JPG) 
5
![example_6](https://user-images.githubusercontent.com/19996897/37919362-46904884-3141-11e8-940f-17434643fdf1.JPG) 
6
![example_7](https://user-images.githubusercontent.com/19996897/37919365-46fa5fda-3141-11e8-8c11-e0b01b41501a.JPG) 
7
![example_8](https://user-images.githubusercontent.com/19996897/37919366-476f02fe-3141-11e8-8a99-8e86e47a4163.JPG) 
8
![example_9](https://user-images.githubusercontent.com/19996897/37919369-47dba8a0-3141-11e8-9dfc-eb5fad70e391.JPG)
9

Softwares and libraries used : Python, Tensorflow, keras, Sklearn.


### Models used for Transfer Learning:
    VGG16
    RESNET50
    
### VGG16 model:
    Built the model using Keras Library.
    Fine tuned the last classification layer for 10 classes and used imagenet weights for training.
    Training Data : 80% of Data
    Testing Data  : 20% of Data
    Batch Size = 32 Epoch = 50
    Training accuracy : 99.8%
    Testing Accuracy  : 92.3%
    
    
### RESNET-50 model
      Built the model using Keras Library.
      Fine Tuned the last classification layer for 10 classes and used the imagenet weights for training.
      Training Data : 80% of Data
      Testing Data  : 20% of Data
      Batch Size = 64 Epochs = 50
      Training accuracy : 99.8%
      Testing Accuracy  : 80.5%
      
      Batch Size :32 Epochs = 50
      Training Accuracy : 99.15%
      Testing Accuracy : 65.15%
    
    
