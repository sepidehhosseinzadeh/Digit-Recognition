# Digit-Recognition
Images of numbers 0 through 9, written in two different kinds of ink, by 6 different people. The hope is to accurately classify the twenty different combinations of numbers and ink.

## Method
I tried two different approaches to solve the problem:
1. I read 10 channels images from the web links, and made a 10 × N × N input, which N is size of the images. Then, a CNN trained on training images (train and test split: 20% of the images are randomly selected for testing and the rest for training). I tried 3 different structures for CNN. The best train accuracy was 99% and test accuracy was 27%.
2. because the algorithm was slow, I tried to build the 10 channels images from the CSV file.

## Codes
- code_0.py, code_1.py, code_2.py: Using second methodology, and different CNN structures.
- code_img.py: Using first methodology.

## How to use the codes
- Changing the path and name_file variables in line number 22 and 23 of the code, to your path
and name of the input file.
- Dependencies to be installed: Tensorflow, numpy, pandas, matplotlib, opencv2, sklearn, keras,
PIL.
(Note: Keras is a high-level neural networks API, written in Python and capable of running on
top of TensorFlow, CNTK, or Theano).
