# Privacy-Preserving Machine Learning

This repository presents a research project focused on ensuring data privacy in machine learning, particularly through the use of various methods such as masking and encryption. The study explores the effectiveness of these techniques on different datasets, including MNIST, Breast Cancer, and Pascal VOC.

## Table of Contents

- [Research Objectives](#research-objectives)
- [Datasets](#datasets)
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


### Network Architectures

Different neural network architectures were employed for various datasets:
- For MNIST, a convolutional neural network (CNN) was used.
- For Breast Cancer classification, a fully connected neural network was implemented.
- MobileNetV2 was utilized for object detection tasks with Pascal VOC.

## Results

The results indicate that while privacy-preserving methods reduce model accuracy, they still maintain a level of effectiveness. Key findings include:
- Models trained on original datasets performed best.
- Masking and encryption techniques led to reduced accuracy, especially at higher masking percentages.
- The combination of masking and encryption resulted in the lowest accuracy among all methods tested.

## Conclusion

The table of results shows that even with these methods, a fairly high accuracy can be achieved.

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
