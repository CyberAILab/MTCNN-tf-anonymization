## MTCNN-tf-anonymization
This is a face detection & anonymization work. Face detection part is a retraining of [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow) with 1~2% improvement on accuracy. Face anonymization is implemented using OpenCV. 

## Dependencies
* Tensorflow 1.6.0
* TF-Slim
* Python 2.7
* Ubuntu 16.04
* Cuda 9.0

## How to test
Inside test folder, run: `python image_test.py`

## How to train
Please refer to the [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow) for training details. 

## Results
![result1.jpg](https://raw.githubusercontent.com/CyberAILab/MTCNN-tf-anonymization/master/result/1.jpg)
![result2.jpg](https://raw.githubusercontent.com/CyberAILab/MTCNN-tf-anonymization/master/result/input.jpg)
![result3.jpg](https://raw.githubusercontent.com/CyberAILab/MTCNN-tf-anonymization/master/result/mask3.png)

## ROC curve on FDDB
![result4.jpg](https://raw.githubusercontent.com/CyberAILab/MTCNN-tf-anonymization/master/result/discROC-compare.png)

## License
MIT LICENSE

## References
[MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)

In addition, this project is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
