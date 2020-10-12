                            INSTALLATION INSTRUCTION TO RUN CNN FOR NETWORK TRAFFIC CLASSIFICATION

NOTE: WE WILL CREATE VIRTUAL ENVIRONMENT AND INSTALL ALL DEPENDENCIES IN THAT SO THAT IT WILL NOT CHANGE/AFFECT YOUR ROOT DIRECTORY, JUST BY DELETEING THIS FOLDER YOU CAN DELETE ITS EXISTANCE.

open windows 'cmd' typing cmd in search (not conda command window) as an administrator



1) Download and install python from> https://www.python.org/downloads/    (if you already have then ignore this step and check its version to be sure)
2) check python version> python -V
3)install pip> python get-pip.py
4) Upgrade pip > python -m pip install -U pip
5) Install virtual environment> pip install virtualenv
6) test installation> virtualenv --version
7) make a folder named 'traffic_classification' in any drive of windows (in my case i made a folder in 'E' drive)
8) Now change the directroy to 'E' drive
(for example if previously it was c:\User>amit then change drive by typing 'cd..' and finaly change to 'E', it will look like 'E:\')
9) In 'E' drive we created a folder named 'traffic_classification' go inside that folder by changeing directory> E:\cd traffic_classification
(now this folder will be your working directory, copy the folder 'trained_model' i have given to you to this folder)

10)create virtual environment repositery>virtualenv cnn_traffic
(this will create a folder with name 'cnn_traffic' inside working reositery 'traffic_classification' and inside 'cnn_traffic' there will be four folders and one license file. )

11) Run the command to activate virtual environment>cnn_traffic\Scripts\activate

(after successful run, cmd window will look like '(cnn_traffic) E:\traffic_classification>' means you have successfully created and activated virtual env and ready to work)

##### Install necessary packages ##########################################################
IGNORE THIS SECTION AS YOU ALREADY INSTALLED ALL PACKAGES

12) Now install Keras by typing: (cnn_traffic) E:\traffic_classification>pip install keras
(this may take few minutes)

13) Install tensorflow using pip: (cnn_traffic) E:\traffic_classification>pip install --upgrade tensorflow
(you can also install tensorflow-gpu but this time i will recommend to follow steps given here)
14) Install numpy using pip : (cnn_traffic) E:\traffic_classification>pip install numpy
15) Install numpy using pip : (cnn_traffic) E:\traffic_classification>pip install pandas
16) Install numpy using pip : (cnn_traffic) E:\traffic_classification>pip install sklearn
#########################################################################################


NOW YOU ARE READY To RUN SCRIPT


directory structure:
 Complete_project
	|
	|-----Train_network
	|-----Test_network

Download 'complete_project' from google drive and Copy both folders (Train_network and Test_network) to folder 'traffic_classification'.



How To run train module---->

17) change directory to train_network i.e go to folder 'train_network':(cnn_traffic) E:\traffic_classification> cd train_network

18)run the python scipt: (cnn_traffic) E:\traffic_classification\train_network>python train_cnn.py

Explanation: A after successfull run a new 'model.h5' file will be created, use this model to test network, this is your CNN trained model. train_cnn_jupyter.ipynb is jupyter file, if you are using jupyter notebook, then use it.




Test generated mode:

17) change directory to test_network i.e go to folder 'test_network':(cnn_traffic) E:\traffic_classification> cd test_network

19) run the python scipt: (cnn_traffic) E:\traffic_classification\test_network>python test_cnn.py -m model.h5




If eveything is okay you will get output saying types of network

HAPPY CODING

-----------------------over---------------------------------------------------------
