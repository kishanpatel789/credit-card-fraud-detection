# Credit Card Fraud Detection

## Description
This is a project deliverable for the course ISYE 6740 Computational Data Analysis which was completed while enrolled in the Master of Science in Analytics program at the Georgia Institute of Technology. The project implements a series of supervised classifiers to predict whether a given credit card transaction is fraudulent or legitimate. The effectiveness of each classifier on the dataset is assessed. A key concern is imbalance as the number of legitimate transactions greatly outweighs the number of fraudulent transactions.

The PDF report presents the summary of the project while the code `credit_card_fraud_detection.py` contains the relevant data manipulation, model creation and validation, and chart generation.

## To Run
1. Download dataset `creditcard.csv` from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Place .csv file in same directory as `credit_card_fraud_detection.py`
3. Ensure 3rd party libraries listed in `requirements.txt` are installed. A virtual environment is recommended.
4. Execute command `python credit_card_fraud_detection.py` to create and assess models.

## Notes
- Depending on the machine, the code may take about 2-5 minutes to run.
- The script will print performance metrics for each classifier to the terminal as well as save relevant charts in the same directory.
