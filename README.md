# BrainTumor_Classification
Brain Tumor Classification

Introduction
Problem Statement

Making use of knowledge on Computer Vision, Image Processing, Pattern Recognition, Computer-aided classification to classify radiology images and histopathology images of brain gliomas, obtained from the same patient, as well as their diagnostic classification label.
Taking into consideration the latest classification of Central Nervous System(CNS) tumors.

-A = Lower grade astrocytoma, IDH-mutant (Grade II or III)
-O = Oligodendroglioma, IDH-mutant, 1p/19q codeleted (Grade II or III)
-G = Glioblastoma and Diffuse astrocytic glioma with molecular features of glioblastoma, IDH-wildtype (Grade IV)

Objective:
The availability of an automated computer analysis tool that is more objective than human readers can potentially lead to more reliable and reproducible brain tumor diagnostic procedures.
To train classification algorithms across the different gliomas classes, so it can correctly identify the type of glioma accurately to reduce redundant efforts of doctors to identify type of glioma and probability of human error in medical examining leading to improve tumor diagnosis and grading process, as well as to enable quantitave studies of the mechanism underlying disease onset and progression. This will facilitate the radiologist in performing diagnosis at a quicker rate.
Why This Specific Project Topic?

Brain tumor/cancer is a very fatal and complex disease with Glioma being the most common type of brain tumor/cancer.
Diagnosis and grading of brain tumors is traditionally done by pathologist.
Pathologist examine tissue section fixed on glass slides under a light microscope; this process continues to be widely applied in clinical settings, it is not scalable to translational and clinical research studies involving hundreds or thousands of tissue specimens.
The detection of a brain tumor at an early stage is a key issue for providing improved treatment. Once a brain tumor is clinically suspected, radiological evaluation is required to determine its location, its size, and impact on the surrounding areas.

 Based on this information the best therapy, surgery, radiation, or chemotherapy, is decided. It is evident that the chances of survival of a tumor-infected patient can be increased significantly if the tumor is detected accurately in its early stage. As a result, the study of brain tumors using imaging modalities has gained importance in the radiology department.





Scope:

Computer-aided classification has the potential to improve tumor diagnosis and grading process, as well as to enable quantitative studies of the mechanisms underlying disease onset and progression.
As Imaging has entered its information era, there has been an increased need to understand and quantify the complex information conveyed by biomedical images. Computational methods offer the potential for extracting diverse and complex information from imaging data, for precisely quantifying it and therefore overcoming limitations of subjective visual interpretation, and for finding imaging patterns that relate to pathologies. They can therefore contribute significantly to automated, reproducible and quantitative interpretations of biomedical images. 


Hardware and Software to be Used:

Hardware :
GPU : To process large data at a comparatively faster rate.
	(Optional )GCP or AWS cloud platform

Software:
Jupyter Notebook & PyCharm: Python IDE
Postman : To test User Interface
Atom : IDE to develop User Interface of the application
ITK-Snap : To navigate three-dimensional medical images
Libraries :
 numpy, nibabel,os, pandas, scipy, seaborn, ipyvolume, PIL, matplotlib
HTML,CSS, .JS, PYTHON, FLASK.
	
Testing Technology:
-Comparing the actual value to predicted value using Test and Validation dataset.


Process Description:

Training Data consist of paired radiology-pathology images and the ground truth classification file. The classification file is a .csv file (training_data_classification_labels.csv) with three columns, where the first column denotes the ID of a case, the second column the classification label and the third column is the age of the subject in days.



A "training_data_pathology_image_info.csv" file that stores the physical pixel size, in microns per pixel, of each histopathology image:

Dataset : Real life dataset requested from University of Pennsylvania. 
Image file: TIFF image file and NII image file

 



NII image file (*.nii) : NII stands for NIfTI – 1 Data Format ( Neuroimaging Informatics Technology Initiative). It’s a standard representation of images and it’s most common used type of analytic file. 

TIFF image file : Tagged Image File Format or TIFF is a computer file format for storing raster graphics images. It supports color depths from 1 to 24-bit and supports both lossy and lossless compression. TIFF files also support multiple layers and pages.

MRI(Magnetic Resonance Imaging) : MRI machine is a large, cylindrical piece of equipment generating a strong magnetic field around the patient. It creates highly detailed pictures of patient’s body’s soft tisse to provide doctor with substantial diagnostic and prognostic information.
Contrast plays important role in MRI. MRI’s are unique to other imaging methods like CT scans and x-rays as they involve the use of gadolinium-based contrast agents(GBCAs) which is a type of MRI dye which goes into patient’s arm intravenously. The contrast medium enhances the image quality and allows the radiologist more accuracy and confidence in their diagnosis.

When the radiologist adds the contrast to your veins, it enhances their visibility of:
•	Tumors
•	Inflammation
•	Certain organs’ blood supply
•	Blood vessels

Side Effects of MRI with contrast may lead to some side effects and safety concerns: 
•	Kidney Issues
•	Joints Inflammation and Irritation
•	Hives
•	Skin Rash
•	Pain at the injection site
•	Dizziness or warmth
•	Nausea and vomiting






To see MRI scans in 3D format we make use of a software called ITK-Snap and initially analyze 4 types of contrast which highlight different areas of brain namely:
•	Flare
•	T1
•	T1ce
•	T2
 










Exploratory Data Analysis


Library used: numpy, nibabel , pandas, scipy.ndimage, seaborn, ipyvolume, PIL,  matplotlib.pyplot

1)	Loading of .csv file : containing subject/patient id, gliomas type and patient age in days.
 

There are in total 221 subjects/patients
 

 

Frequency of gliomas in the dataset
 
Frequency of each type of gliomas
Clearly the dataset is biased towards G type gliomas
 


Bar and line distribution of gliomas
  


2)	Data Analysis for .nii images, for simplicity we will be dealing with only one subject with customer id “CPM19_CBICA_AAB_1” and loading it’s all four types of MRI images using python library called “nibable”.


Loading of image and converting the image into numpy array
 

Shape of .nii image
All the image have a shape of (240,240,155) i.e the .nii 3D image consist of 155 2D images of resolution 240 x 240.
       



 
We concat all 4 image type for input to model. Since 3D CNN is very computationally expensive we make use of either 2D CNN or resnet50 model which takes 2D images as input.
Making use of resnet50 model gives better accuracy as it’s more dense compared to 2D CNN, so we will go ahead with resnet50.

Which Slice of 3D image for input?
We will make use of the concept of patch input where we only put a slice of input instead of the whole 3D image or all 155 2D images, we will find the image where maximum area of tumor is present.
In MRI scans of type t1,t1ce,flair,t2 it increases the intensity of the area where tumor is present, so we will go ahead and make use of intensity to separate the tumor part.

 
   
 
After some trial and error we get to know that after intensity is 450, you get maximum of the tumor present in the MRI scan (exception in 2-3 image, where the threshold intensity was 250 and in some cases 550)

To find the slice of image which contains the maximum area of tumor, after thresholding we make use of sum() function, the slice of image with maximum number of sum() is the slice which got maximum area of tumor.
We make use of function find_max_tumor() which takes  3D image as input and returns the index number which contains the maximum sum for image i.e. portion which contains maximum area of tumor.
 


Creation of dataset, we create X_train,X_test, y_train,y_test for our model where X_train,X_test contain numpy array which contains the concatenated image where y_train,y_test contains the on-hot encoded value which represents the type of the gliomas.
 
 

 

 
 


Resnet50 Model




Karnika





Saving model using JSON
JSON is a simple file format for describing data hierarchically.
Keras provides the ability to describe any model using JSON format with a to_json() function. This can be saved to file and later loaded via the model_from_json() function that will create a new model from the JSON specification.
The weights are saved directly from the model using the save_weights() function and later loaded using the symmetrical load_weights() function.
The example below trains and evaluates a simple model on the Pima Indians dataset. The model is then converted to JSON format and written to model.json in the local directory. The network weights are written to model.h5 in the local directory.
The model and weight data is loaded from the saved files and a new model is created. It is important to compile the loaded model before it is used. This is so that predictions made using the model can use the appropriate efficient computation from the Keras backend.
The model is evaluated in the same way printing the same evaluation score.

User Interface:

For ease of use we made an application with simple functions such as:
Upload Image Button: All four types of images to be uploaded
Submit Button: Which triggers the Machine Learning model to classify the type of tumor
After clicking the classify button, the result of the classification model is shown on the application interface.

Here website connects the user to the machine which will be there on backend server, website allows user to upload the images on the website which can further sent to machine to detect the tumor in the brain images


 


Our client needs to upload images on to the website , and submit it , and then forward it will forwarded to server, which will have the machine learning code, which will Identify whether the client has brain tumor or not.
Functionality Of Application:
•	Give user a glimpse of 3D model of brain
•	Authenticate the user with login details
•	Allow user to upload the brain images
•	Send those images to backend machine learning model 
•	Show the result to the user.

In our application there are mainly 4 html file
-	Home Page
-	Login Page
-	Upload Files Page
-	Result Page

Home Page :
Here user is given glimpse of 3D model of a brain, with interactive functionality of playing with the 3D model by 
i)	Rotating the model
ii)	Increasing the opacity of brain or decreasing it
iii)	Increase/decrease contrast of brain
iv)	Save the image
v)	Increase/decrease the size of your 3D brain model

Clicking on “Let’s get started” button takes you to login page
 


 
Login page :
This page is used to authenticate the user and allow the user to enter the main website where the client can upload the images.


 


Upload Files Page:

User get option to upload 4 types of .nii image file and click on “Submit” button to give input to model for prediction.
After clicking submit button, page then navigates toward “Result Page”.
 

Result Page


Software Used
-	PyCharm
-	JupyterNotebook(Anaconda)
-	Selenium(To automate testing)

Framework and Libraries
-	Flask
-	Selenium
-	Numpy
-	Pandas
-	Matplotlib
-	Tensorflow
-	Keras

Testing


Limitations:

1.	Large image file leading to slower classification rate.



Conclusion

Contribute our work towards current study on  to make use of computer-aided solution to classify the type of Brain Tumor 
Resources:
•	https://www.envrad.com/what-is-an-mri-with-contrast/
•	https://casemed.case.edu/clerkships/neurology/Web%20Neurorad/MRI%20Basics.htm
•	https://medium.com/coinmonks/visualizing-brain-imaging-data-fmri-with-python-e1d0358d9dba
•	https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863141/
•	https://www.hindawi.com/journals/ijbi/2017/9749108/
•	https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
•	https://ugoproto.github.io/ugo_py_doc/viewing_3d_volumetric_data_with_matplotlib/
•	https://www.edureka.co/blog/convolutional-neural-network/
•	https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
•	https://youtu.be/2pQOXjpO_u0
•	https://engmrk.com/kerasapplication-pre-trained-model/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
•	https://engmrk.com/residual-networks-resnets/
•	https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50
•	https://www.researchgate.net/post/Possible_How_Transfer_learning_for_medical_images
•	https://www.researchgate.net/post/Im_fine-tuning_ResNet-50_for_a_new_dataset_changing_the_last_Softmax_layer_but_is_overfitting_Should_I_freeze_some_layers_If_yes_which_ones
•	https://www.kaggle.com/loaiabdalslam/brain-tumor-mri-classification-vgg16
•	https://blog.tensorflow.org/2018/07/an-introduction-to-biomedical-image-analysis-tensorflow-dltk.html
•	https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images/51996037
•	https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/
•	https://www.kaggle.com/albertovilla/applying-restnet50-to-flower-classification
•	https://towardsdatascience.com/transfer-learning-and-image-classification-using-keras-on-kaggle-kernels-c76d3b030649
•	https://www.kaggle.com/risingdeveloper/transfer-learning-in-keras-on-dogs-vs-cats
•	https://www.roytuts.com/python-flask-file-upload-example/
•	https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
•	https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71179
•	https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/
•	https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
