# ChurnPredictorPyTorch
Overview
This notebook, bytorch.ipynb, implements a neural network using PyTorch to predict customer churn based on the "Churn_Modelling.csv" dataset. The goal is to classify whether a customer will exit (1) or stay (0) based on features like credit score, age, balance, and more. The notebook includes data loading, preprocessing, model definition, training, and evaluation.
Dataset

File: Churn_Modelling.csv
Description: A dataset containing customer information for churn prediction.
Columns:
RowNumber: Index of the row (dropped during preprocessing)
CustomerId: Unique customer identifier (dropped)
Surname: Customer surname (dropped)
CreditScore: Customer's credit score
Geography: Customer's country (France, Spain, Germany; label encoded)
Gender: Customer's gender (label encoded)
Age: Customer's age
Tenure: Years with the bank
Balance: Account balance
NumOfProducts: Number of bank products used
HasCrCard: Has credit card (1 or 0)
IsActiveMember: Active member status (1 or 0)
EstimatedSalary: Estimated salary
Exited: Target variable (1 = exited, 0 = stayed)



Requirements

Python 3.x
Libraries:
pandas: For data loading and manipulation
numpy: For numerical operations
sklearn: For preprocessing and train-test split
torch: For building and training the neural network
matplotlib: For visualizing results



Install dependencies:
pip install pandas numpy scikit-learn torch matplotlib

Notebook Structure

Data Loading and Preprocessing

Loads the CSV file using pandas.
Drops irrelevant columns (RowNumber, CustomerId, Surname).
Encodes categorical variables: Geography and Gender are label-encoded.
Splits data into features (X) and target (y).
Performs a train-test split (80% train, 20% test).


Model Definition

Defines a neural network (Model class) using PyTorch:
Input layer: Matches the number of features
Hidden layers: 12 → 20 → 16 → 4 (ReLU activation)
Output layer: 4 → 1 (for binary classification)


Uses sigmoid for binary output.


Training

Sets a random seed for reproducibility.
Uses CrossEntropyLoss (note: current implementation has issues; see Improvements).
Optimizes with Adam (learning rate = 0.01).
Trains for 100 epochs, tracking loss.


Evaluation

Evaluates the model on the test set.
Computes accuracy by comparing predicted classes to true labels.



How to Run

Ensure the Churn_Modelling.csv file is in the same directory as the notebook.
Install required libraries (see Requirements).
Open the notebook in Jupyter or Colab:jupyter notebook bytorch.ipynb


Run all cells sequentially.
Check the console for epoch-wise loss and final test accuracy.

Current Results

Test Accuracy: Approximately 79.85% (may vary due to random seed and data issues).
Note: The loss values printed as -0.0 indicate a potential issue with the loss function or data.

Potential Improvements

Preprocessing:
Use one-hot encoding for Geography instead of label encoding to avoid implying ordinality.
Scale numerical features (e.g., CreditScore, Age) using StandardScaler for better convergence.


Model:
Fix the loss function: Use BCEWithLogitsLoss for binary classification instead of CrossEntropyLoss with manual sigmoid.
Add dropout layers to prevent overfitting.


Training:
Use mini-batches via DataLoader for efficiency and generalization.
Add a validation set to monitor overfitting.


Evaluation:
Include metrics like precision, recall, F1-score, and AUC-ROC, especially if the dataset is imbalanced.
Plot training loss to diagnose training issues.


Reproducibility:
Set random seeds for NumPy and scikit-learn in addition to PyTorch.


Error Handling:
Check for missing values and file existence.



Limitations

The current loss function setup is inconsistent (uses CrossEntropyLoss with a single output and manual sigmoid).
No feature scaling, which can slow convergence or hurt performance.
No validation set, risking overfitting.
Limited metrics; accuracy alone may be misleading if the dataset is imbalanced.

Author

Created on: June 07, 2025
Contact: [Add your contact info, e.g., email or GitHub]

License
This project is unlicensed and provided as-is for educational purposes.
