# Agent based final testing of integrated circuits
This repository contains all means for training and testing a Reinforcement Learning Model for final testing of integrated circuits.\
Developed as part of a bachelor thesis at the Zurich University of Applied Sciences.\
In collaboration with: Institute of Data Analysis and Process Design
# Authors
Moritz Rüegsegger and Nicolas Nyfeler\
contact: moritz.ruegsegger@gmail.com, nyfelnic@students.zhaw.ch
# Contains:
1) Package 'gymnasium-custom': To create a custom Gymnasium Environment that contains a virtual test environment for IC Testing.
2) IC Testing Model: Jupyter Notebook for training and testing the Agent.
3) Docker container to build a docker image which allows the notebook to be run on a virtual machine.
4) requirements.txt: All python requirements that have to met.
# Package gymnasium-custom
## Installation:
1) Add the folder 'gymnasium-custom' to your local Python package folder 'site-packages'.
2) In command prompt (shell), move to the directory 'site-packages'.
3) Execute the following command: pip install -e gymnasium-custom
# Training and Testing the Agent
1) Change the .env file in the 'IC Testing Modell' folder by adding the path of your data (DATA_PATH) and the file name of the specific training data set (TRAINING_FILE).\
Save the .env file.
2) Please follow instructions in jupyter notebook 'RL_IC_Testing.ipynb'
