# Characters Recognition using AlexNet

## Prerequisites

<ul>
<li>Python 3</li>
<li>Keras 2.2.4</li>
<li>Tensorflow 1.13.1</li>
<li>OpenCV 4</li>
<li>Numpy</li>
</ul>

## Usage

Uncompress English.tar.gz into a folder and update the constant DATASET_PATH in train.py accordingly.

To train the model:

```Python
python train.py
```

The model will be saved in the path specified by the constant MODEL_PATH. The model's file name is "trained_model".

To test the trained model on an image, just run:

```python
python predict.py --image name_of_the_image_here
```

Full tutorial is available at http://dangminhthang.com/computer-vision/character-recognition-using-alexnet/
