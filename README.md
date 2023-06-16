# gcn_lstm

### Exploring the Effectiveness of Functional Network Embedding and Sequence Modeling for Time Series Classification

<img width="933" alt="FinalDiagram" src="https://github.com/chapagaisa/gcn_lstm/assets/46834070/a2eeac5c-8ae7-48cb-b5b0-69f4aac30282">

Figure 1: Embedding of functional networks with node attributes using GCN, and sequence embedding using LSTM.

### Dependencies
The models has been tested running in Google Colab which has Python Version of 3.10.12, with following required packages: <br>
1. torch==2.0.1  
2. numpy==1.22.4 
3. pandas==1.5.3 
4. sklearn==1.2.2 
5. torch-scatter==2.1.1 
6. torch-sparse==0.6.17
7. torch-geometric==2.3.1
8. sktime==0.19.1


### Instructions
Step1: Download datasets from https://www.timeseriesclassification.com/dataset.php <br>
Step2: Install dependencies using terminal "pip install -r requirements.txt". If you wish to use .ipynb file in google colab, dependencies are included in the file. <br>
Step3: Edit the name of dataset and path as required with datasets.

### Critical Diagram with alpha 0.07

![cd-diagram1](https://github.com/chapagaisa/gcn_lstm/assets/46834070/11cff7a1-82ae-4e0c-8440-2bb86e2f0043)
