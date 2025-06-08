# Introduction

## Overview
The project implements a **U-ViT** (U-Shaped Vision Transformer) for accurate segmentation of the lungs in Chest X-Ray images. This model aims to assist in the diagnosis of diseases and other abnormalities in the pulmonary domain. This is a recreation of a popular project among learners and is purely done for educational and documentational purposes.   

## Motivation
This model was created for educational value and is open-source for anyone to contribute/investigate. The U-ViT technology used in this model is still relatively new, first developed by [Wang, Z., Cun, X., Bao, J., Zhou, W., Liu, J., & Li, H. (2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf). This model is dedicated as a resource and refrence point for  beginners. 

## Dataset 
Two datasets were used to train this model:
- Oxford_IIT Pet Dataset
- Integrated Lung Segmentation Dataset (Darwin, Montgomery, and Shenzhen)  

Refer to [DATA_CITATION.md](https://github.com/IamArav2012/U_ViT-Lung-Segmentation-Model/blob/main/docs/DATASET_CITATION.md) for additional details. 

## Model Architecture 
This model uses the downsampling and upsampling blocks of a Unet and a ViT encoder as the bottleneck of the architecture.   

### Unet encoder: 
- Convolutional Layer
    - A convolutional layer (```Conv2D```) layer is used in each downsampling block of the architecture. It uses kernels, a set of weights, and slides it directly onto the input image(s), producing feature maps. In my case, the feature maps double every encoder block since it is common practice and the numbers are simple to manipulate. Increasing feature maps too quickly can cause degraded-quality feature maps while increasing feature maps too slowly can result in a model that has unnecessary computational cost. The padding is set to ```'same'``` for consistent dimensions, important for skip connections (```Concatenate()```). A ```l2 regularization``` is added to prevent overfitting as it is a primary method to reset excessive weights. In my model, I use a l2 regularization of ```kernel_regularizer=0.001``` as it is generally a good starting point. For more information about Convolutional layers, visit [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/pdf/1505.04597).
- Max Pooling  
    - A Max Pooling (```MaxPooling2D()```) layer uses a technique called "pooling" to downsample images by a certain amount, contingent on hyperparameters like ```pool_size``` and ```strides```. In the model, a ```pool_size=2``` is used and ```strides``` is not explictly defined since, by default, ```strides``` is set equal to the ```pool_size``` unless defined otherwise. I use a ```MaxPooling2D()``` layer in each downsampling block, and the corresponding ```Conv2dTranspose()``` and ```Concatentate()``` layers. Max Pooling is significant because it allows convolutional layers to effectively have a "larger" receptive field since they extract the maximum value from each individual pool, allowing deeper pattern recognition as model depth increases. To learn more, visit the classic novel paper [*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). 

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
    -  Layer Normalization is a version of ```Batch_Normalization``` that was first used in [*Layer Normalization*](https://arxiv.org/pdf/1607.06450).Layer Normalization works identically to batch_normalization except that it calculates mean and standard deviation for each feature and adjusts them seperately instead of performing an operation with averaged values among the entire batch. It is important to note that the features are not directly divided by standard deviation, but the square root of it. In my case, I used ```epsilon = 10e-6`` to avoid a divison by 0 error and to stabalize gradients. For more about batch_normalization visit the **BatchNormalization** section below. 
 
3. MultiHeadAttention  
    - ```MultiHeadAttention``` is used in all modern transformer to help generalization and quicken convergence. It was first used in [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762). ```layers.MultiHeadAttention(num_heads=attention_heads, key_dim=projection_dims, dropout=0.1)``` in the code for modularity and to avoid hard-coding. Dropout in this stage is ettential to combat overfitting. 
4. Add  
    - This is a residual layer used for preventing [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). The ```Add()``` is used after all major layers like the Mlp and MultiHeadAttention. This layer was dervied from the ResNET architectiure which was first implemented in [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385).
5. Mlp 
    - This is a version of a FFN((Feedforward_Neural_Network)[https://en.wikipedia.org/wiki/Feedforward_neural_network]). It uses dense and dropout layers to expand and condense the data for optimal feature extraction. This also ensures shape compatibility.

### U-Net Decoder
1. Conv2DTranspose  
    - ```Conv2DTranspose()``` uses **Transposed Convolution**, also known as **Deconvolution**, to effectively reverse the effect of convolutional layers. This is a ```Conv2d()```and ```UpSampling()``` inspired layer which ettenitialy uses a learned upsampling operation using kernels. Mathematically, it reverses the effect of a Conv2D by "spreading out" the values. This is superior to ```UpSampling``` because it uses learned weights for the kernels just like the ```Conv2D``` layers. This mechanism was first developed in [*Fully Convolutional Networks for Semantic Segmentation*](https://arxiv.org/pdf/1605.06211v1). Furthermore, to fully understand this concept the paper [*A Guide to Convolution Arithmetic for Deep Learning*](https://arxiv.org/pdf/1603.07285) is also recommended.  
 
2. Concatenate  
    - The function ```Concatenate``` is used as [Skip Connections](https://medium.com/@preeti.gupta02.pg/understanding-skip-connections-in-convolutional-neural-networks-using-u-net-architecture-b31d90f9670a) to refactor the encoder's feature maps into consideration. The idea of skip connections were first used in the original U-Net paper [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/pdf/1505.04597). 

3. ```Conv2D(1, 1, activation='sigmoid')```
    - This layer uses a **1 by 1 kernel** and the ````'sigmoid'``` activation to generate pixel by pixel mask predictions. The 1 by 1 kernal ensures the mask has the same spatial dimensions as the input image since it has already been upsampled to appropriate dimensions ```(128 by 128)```. Additionally, the ```sigmoid``` activation function, in this case, mainly saturates the output into the a range from 0 to 1 to make the mask interpertable.

### Universal Layers  
These layers will be more thoroughly explained as they are universal to almost all deep-learning neural networks. 

1. Batch Normalization  
    - ```BatchNormalization()``` uses a process of subtracting the mean of the data and dividing by the square root of the standard deviation. Then the output is scaled and shifted, meaning learnable multiplication and addition operations occur. This essentialy eliminates a problem in neural networks called [Internal Covariate Shift](https://en.wikipedia.org/wiki/Batch_normalization). This idea was orignally introduced by the paper, [*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://arxiv.org/pdf/1502.03167.). Batch Normalization has inspired many other types of Normalization layers including LayerNorm, InstanceNorm, GroupNorm, and SpectralNorm.  
2. Activation  
    - An activation function, as the name suggests, applies an activation to the input. There are innumerable activation functions but the ones used in my model are ```'gelu'``` and, primarily, ```'relu'```. The ReLU funciton is characterized by the equation ```f(x) = max(0, x)```, and it works to remove negative values from the network while preserving the positive values. This function was a major breakthrough in modern machine learning since previously tanh(x) was used which saturated postive values, not just negative ones. The ```GELU``` activation function also works to saturate negative values but rather than eliminating them completely, it incorperates small negative values in the neural network, helping very deep models like [**BERT**](https://huggingface.co/docs/transformers/en/model_doc/bert) and [**GPT**](https://zapier.com/blog/what-is-gpt/). In my model, ReLU is implemented everywhere since it is computationally inexpensive and considered good practice. In contrast, GELU is only used once, in the **mlp** unit of the transformer layer since ```mlp()``` is vital to extract the most optimal features.  
3. Dropout  
    - ```Dropout()``` is a techinique introduced by [*A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). It deactivates a certain percentage (```dropout_rate```) of activations in a layer forcing the model to learn redundant and comphrehensive representations. This is a anti-overfitting technique that improves generalization, similar to ```regularization``` but utilizing nuanced methodology. I use dirrent dropout rates in my ```pass_through_vit()```,  ```build_uvit()```, and ```mlp()``` ranging from 0.1 (10%) to 0.35 (35%) depending on how crucial the operation is for overfitting.  

## Training Setup 
### Pretrain_segmentator.py
#### Data Preparation
The Oxford_IIT dataset is loaded from ```tensorflow_datasets``` by using the ```.load()``` attribute. This script utilizes the ```preprocess()``` and ```augment()``` functions to prepare the Oxford-IIT data for training, validation, and testing. The data is split using ```train_test_spilt()``` and manually converted into a ```tf.data.Dataset``` for faster training and CPU/GPU optimization. These datasets are independently batched, prefetched, and  augmented according to best practices. 

#### Pachify and PatchEncoder
This model uses custom classes named ```Pachify``` and ```PatchEncoder```to integrate the output of the U-Net's downsampling block with the ViT architecture. These classes are decorated with ```@register_keras_serializable()``` for seamless loading into the ```Lung_Segmentator.py``` script. The Pachify class uses ```tf.extract_patches``` to extract patches from the feature maps, and it also reshapes the input from ```(batch_size, img_height, img_width, feature_maps)``` to ```(batch_size, img_size, patch_dims).``` It is important to note that ```feature_maps``` and ```patch_dims``` remain unchanged in value but get renamed for intuition and conceptual clarity. The class ```PatchEncoder``` performs a linear transformation and provides ```positional_embeddig``` for spatial context. It converts the 'pixel values' into resized vectors for ViT processing, and then it adds the positional embeddings directly to the vectors. The positional embeddings are derived from a learned ```layers.Embedding``` layer that takes indices as its input.    

### Lung_Segmentator.py
This script uses a ```load_and_preprocess()``` function and an augment function, similar to ```pretrain_segmentator script.``` The main difference is that the ```os``` module is being used to directly load from the hard drive, and we are using stronger augmentation ([Gaussian Distribution](https://en.wikipedia.org/wiki/Normal_distribution)). This script is using ```tf.keras.models.load_model``` to directly load the ```pretrained_segmentator_weights.keras``` file as initialized weights. The ```LEARNING_RATE``` has been decreased for fine-tuning, and ```Early_Stopping``` is similarly implemented. A ```combined_loss``` function is used which combines a weighted sum of ```dice_loss``` and ```binary_crossentropy``` from a balanced ratio of 0.65 to 0.35 respectively.   

## Results and Analysis