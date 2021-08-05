# Face-Recognition


&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
<p align="center"> 
This work is entirely inspired from the following repository : https://github.com/wondonghyeon/face-classification.  
Wondonghyeon, thank you for your great work !
</p>

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;


We have developed a DNN to classify faces according to their gender (male/female). We used Wondonghyeon’s work, Python3 and Keras/Tenserflow. Let's test and re-train this model !           |  
:-------------------------:

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

---------------------------------------------------------------
&nbsp;


# Table of contents
- [Tree structure](#id-section1)
- [Packages required](#id-section2)
- [Instructions](#id-section3)
- [Dataset](#id-section4)
- [The project (step by step)](#id-section5)
  - [Model Testing](#id-section6)
  - [Accuracy](#id-section7)
  - [Model Training](#id-section8)
- [What about Prométhée ?](#id-section9)




&nbsp;
&nbsp;
<div id='id-section1'/>

## Tree structure 

(mettre le tree)



Here is our project file, « GenderDetection ». You will find on it :
- a code opensource file named **GenderDectection**, containing all useful codes for the project
- a **results** file, to store results of predictions (and ? Comparisons etc ?)
- an empty file named **Dataset**, which you will fill later (see [Instructions](#id-section3))

&nbsp;
&nbsp;
<div id='id-section2'/>

## Packages required

This project has been developed with Python3, using Keras/TenserFlow
- [Python3](https://realpython.com/installing-python/)
- [Keras and Tenserflow](https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/)

Here are listed the packages required for the project :
- [dlib](http://dlib.net/)
- [face_recognition](https://github.com/ageitgey/face_recognition/)
- [Numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

&nbsp;
&nbsp;
<div id='id-section3'/>

## Instructions

Let’s begin ! Once you have installed all packages, download the **GenderDetection** file. Remember
its path (ou donner commande pour le retrouver? Juste sous linux?).
As an exemple, my path is : home/user1/promethee/.

- If you want to use our data : go to [Dataset](#id-section4), follow instructions, and dezip it on **GenderDetection** file. 
- If you prefer using your own data : (?)

On your terminal, use the cd command and run cd + your GenderDetection path. In my case :
&nbsp;
  
`cd home/user1/promethee/GenderDetection/` 

You should find here all the folders shown in the tree structure : run `ls -lah` to check it. 

&nbsp;
&nbsp;
<div id='id-section4'/>

## Dataset

Our **Dataset** folder can be provided on demand. As shown on the scheme, this file is divided into
four other ones :

- **test**, containing 65554 images from wondonghyeon’s dataset to test the model
- **train**, containing 131141 images from wondonghyeon’s dataset to train the model 
- **test2**, containing 12896 images from our dataset to test the model
- **train 2**, containing 25791 images from our dataset to train the model

You can ask for access by clicking here : (link). 
Meanwhile, you can discover the project ([The project (step by step)](#id-section5)) and Promethee’s concept([What about Prométhée ?](#id-section9))

&nbsp;
&nbsp;
<div id='id-section5'/>

---------------------------------------------------------------
&nbsp;

## The project (step by step)

&nbsp;
<div id='id-section6'/>

### Model testing 

_pred.py_ script is directly inspired from Wondonghyeon’s project. Its purpose is to test the _face_model.pkl_ 
model, which detects faces on the image. Now please run the following command :

`python3 pred.py`


This script takes images from the **test** dataset (by Wondonghyeon), generates a csv by image as the
one below, and stores it on the **results** folder):


![image](https://user-images.githubusercontent.com/88309709/128307869-b716dc44-3897-4baa-9fcd-a60ee83fd8b2.png)

Please focus on the "Male" indication : close to 1 means "man" , close to 0 means "woman". Here, _pred.py_ detects a woman.
On [Accuracy](#id-section7), we will see how accurate is this prediction. 

&nbsp;


To test images from our dataset (in **test2**), please run :

`python3 pred.py --img_dir ../Dataset/test2--output_dir results2`

Similarly, results will be stored on the **result2** folder.


&nbsp;
<div id='id-section7'/>

### Accuracy

Let’s compare our results with reality. We will focus on gender : is our model enable to find gender
when analyzing a face ?

_List_attr_celeba.txt_ file contains real features for each picture in the **test** folder (Wondonghyeon’s
pictures). For each image, features are displayed on the same line. When looking at Column n°21 ("Male"): -1 is used for describing a woman, and 1 for 
describing a man. To compare it with **results** content, please run the following command :

`python3 comput_acc.py`

The script displays an accuracy percentage. We get "acc :98%", and you? 

&nbsp;

For obtaining an accuracy percentage using our dataset, please run : 

`python3 comput_acc.py --input_dir results2 --input_label ../Dataset/coco_open.txt`

This script uses our _coco_train.txt_ file, containing features for each picture in the **test2** folder. Features are compared with **results2** content. We get "acc : 88%", what about you? 


&nbsp;
<div id='id-section8'/>

###  Model training 

Let's move to our model training. Before running the command, some explanations : 

For each picture, our _gen_train_data.py_ script generates a 128 octets' signature (stored in _X.npy_) and a gender label (stored in _Y.npy_).
The _train.py_ script works with **train** and **train2** folders. It uses _X.npy_ and _Y.npy_ to run our DNN. An accuracy is given
. To see it by yourself, please run the following command : (could take a long time)

`python3 train.py`

What percentage are you getting? We roughly get 98%. 
In order to improve it, lets first modify the rate (_train.py_, line 26). 

&nbsp;
&nbsp;
<div id='id-section9'/>

---------------------------------------------------------------
&nbsp;

# What about Prométhée? 

&nbsp;
[Prométhée] is an association, aiming to bring together : 
+ qualified volonteers, loving to work on IA (especially on images classification)
+ volonteers willing to support useful and ethical projects

&nbsp;

Want to join us/ to know more about Prométhée ? Please click on there : (link promethee page Fb?)



