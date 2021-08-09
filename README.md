# Face-Recognition


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
&nbsp;

<p align="center"> 
This work is inspired from the following repository : https://github.com/wondonghyeon/face-classification.  
Wondonghyeon, thank you for your great work !
</p>

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
  - [Model testing : your images](#id-section6)
  - [Model testing : our dataset](#id-section7)
  - [Model Training](#id-section8)
- [What about Prométhée ?](#id-section9)




<div id='id-section1'/>
&nbsp;   

## Tree structure 

&nbsp;
&nbsp;
![image](https://user-images.githubusercontent.com/88309709/128673954-29e9922d-b7d6-40f1-9fac-9563cb84a8b2.png)
&nbsp;
&nbsp;





Here is our project file, « GenderDetection ». You will find on it :
- a code opensource file named **GenderClassification**, containing all useful codes for the project
- a **train** file, to store all versions of your training
- a file named **Dataset**, which you can retrieve on demand (see [Instructions](#id-section3))

&nbsp;

<div id='id-section2'/>
&nbsp;   

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

<div id='id-section3'/>
&nbsp;   

## Instructions

Let’s begin ! Once you have installed all packages, download the **GenderDetection** file. Remember
its path.
As an exemple, my path is : home/user1/promethee/.

- If you want to use our data : go to [Dataset](#id-section4), follow instructions, and dezip it on **GenderDetection** file. 
- If you prefer using your own data : put your image(s) (format .jpg) on **..GenderDetection/Dataset/mypicture_data** directory. 

On your terminal, use the cd command and run cd + your GenderDetection path. In my case :
&nbsp;
  
`cd home/user1/promethee/GenderDetection/` 

You should find here all the folders shown in the tree structure : run `ls -lah` to check it. 


&nbsp;
<div id='id-section4'/>
&nbsp;   

## Dataset

Our **Dataset** folder can be provided on demand. As shown on the scheme, this file is divided into
four other ones :

- **test**, containing (65554) images from wondonghyeon’s dataset to test the model (thus 30% of images are used for validation)
- **train**, containing (131141) images from wondonghyeon’s dataset to train the model 
- **test2**, containing (12896) images from our dataset to test the model (here to, 30% of images are used for validation)
- **train2**, containing (25791) images from our dataset to train the model
- **mypicture_data**, containing your own image(s) if wished



You can ask for access by clicking here : contact@perception.onl. 
Meanwhile, you can discover the project ([The project (step by step)](#id-section5)) and Promethee’s concept ([What about Prométhée ?](#id-section9))

&nbsp;

<div id='id-section5'/>

---------------------------------------------------------------
&nbsp;

## The project (step by step)


<div id='id-section6'/>
&nbsp;   

### Model testing : your images

_test.py_ script is inspired from Wondonghyeon’s project. Its purpose is to test our model, _default_model.h5_, which detects and caracterizes faces on the image. If you have added your own images on **mypicture_data**, please run the following command : (else, move to [Model testing : our dataset](#id-section7))

`python3 test.py`

&nbsp; 

>If you get this kind of error : 
>#File "/home/user/.local/lib/python3.6/site-packages/sklearn/model_selection/_split.py", line 25, in <module>
>    from scipy.misc import comb
>ImportError: cannot import name 'comb'  
>run :  
>nano + the directory of the file (in my case : nano /home/user/.local/lib/python3.6/site-packages/sklearn/model_selection/_split.py)
>then replace « from scipy.misc import comb » by « from scipy.special import comb ».
>Redo for each error message. 
>
>Why ? For deprecated scipy version, « comb » was on the « misc » module :it is now on the « misc » one. 
  
&nbsp;   


This script takes images from the **mypicture_data** dataset. 
It displays your images, and adds gender prediction on the title : you can read "It's a Man !" or "It's a Woman !". Let's see an example :
&nbsp; 

  
  Your image            |  What test.py displays
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/88309709/128505009-91ec72a1-2266-494d-89bb-7805417e893b.jpg" width="400">  |  <img src="https://user-images.githubusercontent.com/88309709/128504890-d7a4aa0b-da04-4670-a634-e2e88bcd3de5.png" width="400"> 

&nbsp; 
  
Did _test.py_ succeeded on your image(s)?  

<div id='id-section7'/>
&nbsp;   

### Model testing : our dataset 
  
_gen_val_data.py_ script uses **test2** dataset. Then it generates a 128 octets signature and a gender label, for each image.  
We are going to evaluate our model's accuracy on this dataset : please run  
  
&nbsp;
  
`python3 gen_val_data.py`  
And then :   
`python3 evaluate.py`
 
&nbsp;   
  
_evaluate.py_ returns:
- the number of errors (how many "Women" images have been labeled as "Men" ? And vice-versa). 
- the accuracy (relative number of good predictions, in %)
  
  
Normally you should get an accuracy of approximately 98%: let's improve it !  


&nbsp;
<div id='id-section8'/>

 &nbsp;   
  
  
### Model training

_gen_train_data.py_ script uses **train** and **train2** dataset(s). Then it generates a 128 octets signature and a gender label, for each image.  
We are going to improve our model's accuracy on these datasets : please run  
  
&nbsp;
  
`python3 gen_train_data.py`   
And then:   
`python3 train.py`
 
&nbsp;   
  
_train.py_ returns:
- the accuracy of our model 
- its target loss
  
This trained model is stored in **../GenderClassification/train**. To use it next time, please replace our "GenderDetection_v1.h5" by this one.
  
  
Which accuracy are you getting? 





&nbsp;
&nbsp;
<div id='id-section9'/>

---------------------------------------------------------------
&nbsp;

# What about Prométhée? 

&nbsp;
> **Prométhée** is an association, aiming to bring together : 
> + qualified volonteers, loving to work on IA (especially on images classification)
> + volonteers willing to support useful and ethical projects

&nbsp;

Want to join us/ to know more about Prométhée ? Please send us an email !



