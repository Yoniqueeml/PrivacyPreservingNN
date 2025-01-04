# Privacy-Preserving Machine Learning

This repository presents a research project focused on ensuring data privacy in machine learning, particularly through the use of various methods such as masking and encryption. The study explores the effectiveness of these techniques on different datasets, including MNIST, Breast Cancer, and Pascal VOC.

## Table of Contents

- [Research Objectives](#research-objectives)
- [Datasets](#datasets)
- [Network Architectures](#network-architectures)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Google Colab](#google-colab)
- [References](#references)

## Research Objectives

The main objectives of this research include:

- Analyzing existing approaches to data privacy in machine learning.
- Evaluating the impact of privacy-preserving techniques on model performance.
- Exploring the balance between data privacy and model accuracy.

## Datasets

The following datasets were used in this research:

1. **MNIST**: A widely-used dataset containing 70,000 images of handwritten digits.
2. **Breast Cancer**: A dataset with 569 samples, each described by 30 attributes related to breast tumor characteristics.
3. **Pascal VOC**: A dataset for visual object classification, containing over 11,000 annotated images across 20 object categories.

## Network Architectures

Different neural network architectures were employed for various datasets:

### MNIST - Convolutional Neural Network (CNN)

![CNN Architecture](https://github.com/Yoniqueeml/PrivacyPreservingNN/tree/master/arch_images/mnist.png)

### Breast Cancer - Fully Connected Neural Network

![Fully Connected Network Architecture](https://github.com/Yoniqueeml/PrivacyPreservingNN/tree/master/arch_images/breast_cancer.png)

### Pascal VOC - MobileNetV2

![MobileNetV2 Architecture](https://github.com/Yoniqueeml/PrivacyPreservingNN/tree/master/arch_images/mobilenet.png)

## Masking and Encryption Examples

Below is an example of data :

![Mnist example](https://github.com/Yoniqueeml/PrivacyPreservingNN/tree/master/arch_images/mnist_examples.png)
![PascalVOC example](https://github.com/Yoniqueeml/PrivacyPreservingNN/tree/master/arch_images/pascalvoc_example.png)

## Results
The following table summarizes the results of training models on various datasets using different methods:
m - masked, e - encryption

| Dataset       | Method      | Masking (%) | Accuracy (%) | Epochs |
|---------------|-------------|-------------|--------------|--------|
| MNIST         | -           | -           | 98.93        | 10     |
| MNIST         | m           | 20          | 98.27        | 10     |
| MNIST         | m           | 50          | 97.55        | 10     |
| MNIST         | e           | -           | 96.26        | 10     |
| MNIST         | m + e       | 20          | 94.28        | 10     |
| MNIST         | m + e       | 50          | 88.24        | 10     |
| Breast Cancer | -           | -           | 96.49        | 20     |
| Breast Cancer | m           | 20          | 93.86        | 20     |
| Breast Cancer | m           | 50          | 89.47        | 20     |
| Breast Cancer | e           | -           | 90.35        | 20     |
| Breast Cancer | m + e       | 20          | 85.09        | 20     |
| Breast Cancer | m + e       | 50          | 81.58        | 20     |
| Pascal VOC    | -           | -           | 95.97        | 10     |
| Pascal VOC    | m           | 30          | 93.76        | 10     |
| Pascal VOC    | m + e       | 30          | 87.71        | 10     |
| Pascal VOC    | m + e       | 30          | 90.83        | 50     |

The results indicate that while privacy-preserving methods reduce model accuracy, they still maintain a level of effectiveness.

## Conclusion

This research highlights the importance of developing effective mechanisms to protect data privacy in machine learning while maintaining model performance. The findings underscore the need for a balance between confidentiality and accuracy.

## Future Work

Future research may focus on optimizing model architectures and exploring more advanced encryption and masking techniques to minimize accuracy loss while ensuring robust data privacy.

## Google Colab

You can access the MobileNet implementation and experiments through this [Google Colab notebook](https://colab.research.google.com/drive/1CWMkiVRJJ3ZaNylNCFYXs5D65IV6rJTo?usp=sharing).

## References

1. [Data Privacy in Machine Learning Systems](https://www.sciencedirect.com/science/article/pii/S0167404823005151#se0010)
2. [What is Differential Privacy?](https://www.statice.ai/post/what-is-differential-privacy-definition-mechanismsexamples#:~:text=Differential%20privacy%20is%20a%20mathematical,any%20individual%20in%20the%20dataset.)
3. [Synthetic Data Overview](https://www.edps.europa.eu/press-publications/publications/techsonar/synthetic-data_en#:~:text=Synthetic%20data%20is%20artificial%20data,undergoing%20the%20same%20statistical%20analysis.)
4. [Federated Learning and Differential Privacy](https://blog.openmined.org/untitled-3/)
5. [Understanding Differential Privacy](https://www.unite.ai/ru/what-is-differential-privacy/)
6. [High-Accuracy Differentially Private Image Classification](https://deepmind.google/discover/blog/unlocking-high-accuracy-differentially-private-image-classification-through-scale/)
7. [Adaptive Optimizers with Differential Privacy](https://twitter.com/litian0331/status/1548867175502323712)
8. [Privacy-Preserving Machine Learning with Fully Homomorphic Encryption](https://arxiv.org/abs/2106.07229)
9. [Homomorphic Encryption in Machine Learning](https://blog.openmined.org/ckks-homomorphic-encryption-pytorch-pysyft-seal/)
