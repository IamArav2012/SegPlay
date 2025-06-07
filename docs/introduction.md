# Introduction

## Overview
The project implements a U-ViT (U-Shaped Vision Transformer) for accurate segmentation of the lungs in Chest X-Ray images. This model aims to assist in the diagnosis of diseases and other abnormalities in the pulmonary domain. This is a recreation of a popular project among learners and is purely done for educational and documentational purposes.   

## Motivation
This model was created for educational value and is open-source for anyone to contribute/investigate. The U-ViT technology used in this model is still relatively new, first developed by [Wang, Z., Cun, X., Bao, J., Zhou, W., Liu, J., & Li, H. (2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf). This model is dedicated as a resource and refrence point for  beginners. 

## Dataset 
Two datasets were used to train this model:
- Oxford_IIT Pet Dataset
- Integrated Lung Segmentation Dataset (Darwin, Montgomery, and Shenzhen)  

Refer to [DATA_CITATION.md](https://github.com/IamArav2012/U_ViT-Lung-Segmentation-Model/blob/main/docs/DATASET_CITATION.md) for additional details. 

## Model Architecture 
This model uses a encoder and decoder of a unet and a vit_encoder inspired block for the bottleneck. 
### Unet encoder: 
- convution layer
- batchnorm
- activation
- droopout
- max_pooling  
### ViT bottleneck
The exact architecture is: 
```
Pachify()
PatchEncoder()

One Head:
    LayerNormalization()
    MultiHeadAttention()
    Add()
    LayerNormalization()
    Mlp()
    Add()
```

1. Patchify and PatchEncoder  
    - These are the explained thoroughly in the **Pachify and PatchEncoder** section below. 
2. LayerNormalization 
    -  Layer Normalization is a version of ```Batch_Normalization``` that was first used in [*Layer Normalization*](https://arxiv.org/pdf/1607.06450).Layer Normalization works identically to batch_normalization except that it calculates mean and standard deviation for each feature and adjusts them seperately instead of performing an operation with averaged values among the entire batch. In my case, I used ```epsilon = 10e-6`` to avoid a divison by 0 error and to stabalize gradients. For more about batch_normalization visit the **BatchNormalization** section below. 
 
3. MultiHeadAttention  
    - ```MultiHeadAttention``` is used in all modern transformer to help generalization and quicken convergence. It was first used in [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762). ```layers.MultiHeadAttention(num_heads=attention_heads, key_dim=projection_dims, dropout=0.1)``` in the code for modularity and to avoid hard-coding. Dropout in this stage is ettential to combat overfitting. 
4. Add  
    - This is a residual layer used for preventing [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). The ```Add()``` is used after all major layers like the Mlp and MultiHeadAttention. This layer was dervied from the ResNET architectiure which was first implemented in [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385).
5. Mlp 
    - This is a version of a FFN((Feedforward_Neural_Network)[https://en.wikipedia.org/wiki/Feedforward_neural_network]). It uses dense and dropout layers to expand and condense the data for optimal feature extraction. This also ensures shape compatibility.

### U-Net Decoder
1. Conv2DTranspose  
    - 
2. Concatenate  
    - 
3. BatchNormalization  
    - It is important to note that it is not directly divided by standard deviation but the square root of it. 
4. Activation  
    - 
5. Dropout  
    - 

The last is activation is sigmoid with a 1 by 1 kernel to retain original image dimensions.

## Training Setup 
### Pretrain_segmentator.py
#### Data Preparation
The Oxford_IIT dataset is loaded from ```tensorflow_datasets``` by using the ```.load()``` attribute. This script utilizes the ```preprocess()``` and ```augment()``` functions to prepare the Oxford-IIT data for training, validation, and testing. The data is split using ```train_test_spilt()``` and manually converted into a ```tf.data.Dataset``` for faster training and CPU/GPU optimization. These datasets are independently batched, prefetched, and  augmented according to best practices. 

#### Pachify and PatchEncoder
This model uses custom classes named ```Pachify``` and ```PatchEncoder```to integrate the output of the U-Net's encoder with the ViT architecture. These classes are decorated with ```@register_keras_serializable()``` for seamless loading into the ```Lung_Segmentator.py``` script. The Pachify class uses ```tf.extract_patches``` to extract patches from the feature maps, and it also reshapes the input from ```(batch_size, img_height, img_width, feature_maps)``` to ```(batch_size, img_size, patch_dims).``` It is important to note that ```feature_maps``` and ```patch_dims``` remain unchanged in value but get renamed for intuition and conceptual clarity. The class ```PatchEncoder``` performs a linear transformation and provides ```positional_embeddig``` for spatial context. It converts the 'pixel values' into resized vectors for ViT processing, and then it adds the positional embeddings directly to the vectors. The positional embeddings are derived from a learned ```layers.Embedding``` layer that takes indices as its input.    

### Lung_Segmentator.py
This script uses a ```load_and_preprocess()``` function and an augment function, similar to ```pretrain_segmentator script.``` The main difference is that the ```os``` module is being used to directly load from the hard drive, and we are using stronger augmentation ([Gaussian Distribution](https://en.wikipedia.org/wiki/Normal_distribution)). This script is using ```tf.keras.models.load_model``` to directly load the ```pretrained_segmentator_weights.keras``` file as initialized weights. The ```LEARNING_RATE``` has been decreased for fine-tuning, and ```Early_Stopping``` is similarly implemented. A ```combined_loss``` function is used which combines a weighted sum of ```dice_loss``` and ```binary_crossentropy``` from a balanced ratio of 0.65 to 0.35 respectively.   

## Results and Analysis
