# Agent based final testing of integrated circuits
This repository contains all means for training and testing a Reinforcement Learning Model for final testing of integrated circuits.\
Developed as part of a bachelor thesis at the Zurich University of Applied Sciences.\
In collaboration with: Institute of Data Analysis and Process Design
# Authors
Moritz Rüegsegger and Nicolas Nyfeler
contact: moritz.ruegsegger@gmail.com, nyfelnic@students.zhaw.ch
# Contains:
1) Package 'gymnasium-custom': To create a custom Gymnasium Environment that contains a virtual test environment for IC Testing.
2) IC Testing Model: Jupyter Notebook for training and testing.
3) requirements.txt: All python requirements that have to met.
# Package gymnasium-custom
## Installation:
2) Add the folder 'gymnasium-custom' to your local Python package folder 'site-packages'.
3) In command prompt (shell), move to the directory 'site-packages'.
4) Execute the following command: pip install -e gymnasium-custom
# Training and Testing the Agent
1) Change the .env file, located in the folder 'IC Testing Modell'. Add the path of your training and test data (DATA_PATH) and the file name of the specific training data set (TRAINING_FILE).\
Save the .env file.
2) Please follow instructions in jupyter notebook 'RL_IC_Testing.ipynb'
