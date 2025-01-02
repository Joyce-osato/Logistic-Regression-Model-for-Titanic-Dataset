# Logistic-Regression-Model-for-Titanic-Dataset
## Introduction
I am Ehimwenma Joyce Osatohamwen, a tech student at PSP analytics. For my first machine learning project, I used the famous Titanic dataset from Kaggle. The goal of this project was to build a logistic regression model to predict survival outcomes on the Titanic, gaining insights into the factors that influenced survival. Logistic regression is a powerful yet straightforward algorithm that is particularly well-suited for binary classification tasks like this one.

## Data Background
The Titanic dataset is one of the most well-known datasets in machine learning. It contains information on passengers such as age, gender, class, ticket fare, and whether they survived. The dataset is hosted on Kaggle and includes both training and testing datasets.
The main columns in the dataset include:

-	PassengerId: A unique identifier for each passenger.
-	Survived: Survival status (0 = No, 1 = Yes).
-	Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
-	Name: Passenger’s name.
•	Sex: Gender.
•	Age: Age in years.
•	SibSp: Number of siblings/spouses aboard the Titanic.
•	Parch: Number of parents/children aboard the Titanic.
•	Ticket: Ticket number.
•	Fare: Passenger fare.
•	Cabin: Cabin number.
•	Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
Data Analysis Process
To effectively analyze and model the data, I followed these steps:
1. Data Gathering
The Titanic dataset was downloaded from Kaggle, containing two CSV files: train.csv and test.csv. These files provided labeled and unlabeled data, respectively. The labeled training set was used to train the logistic regression model, while the test set was used to evaluate its performance.
________________________________________
2. Business Question
The central question for this project was:
Can we predict whether a passenger survived based on available features?
Additional sub-questions include:
•	How did factors like gender, class, and age impact survival?
•	Can we achieve a model accuracy of at least 75%?
________________________________________
3. Data Assessment and Formatting
Initial Data Review
The dataset was loaded using Python’s Pandas library to inspect its structure and identify missing or inconsistent values:
 
Handling Missing Values
•	Age: Missing values were imputed using the median age of passengers. 
•	train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
•	Embarked: Rows with missing Embarked values were dropped as there were only two such rows. 
•	train_data.dropna(subset=['Embarked'], inplace=True)
•	Cabin: Excluded from analysis due to a high proportion of missing values.
Encoding Categorical Variables
Categorical variables like Sex and Embarked were converted into numerical values using categorical encoding:
 

4. Data Preparation for Modeling
Splitting the Data
The dataset was split into features (X) and target (y), and further divided into training and validation subsets:
 
6. Model Building and Evaluation
Model Training
A logistic regression model was trained using Scikit-learn:
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
Model Evaluation
Performance was evaluated using accuracy, a confusion matrix, and classification report:
 
Recommendations:
•	In future scenarios, prioritize resources for vulnerable groups like women and children during emergencies.
•	Improve safety measures in third-class areas to increase survival rates for economically disadvantaged passengers.
Conclusion
For this machine learning project, I successfully built a logistic regression model to predict Titanic survival. This project provided hands-on experience with data cleaning, feature engineering, and model evaluation. The insights derived from the data analysis demonstrate the importance of demographic and socio-economic factors in survival outcomes. Moving forward, I plan to explore more complex models like Random Forests and Gradient Boosting to improve prediction accuracy.

