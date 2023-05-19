# Airbnb Price Prediction in San Francisco
## Using Word2Vec, PCA + Linear Regression, Random Forest, Gradient Boosting, KNN

### Background
Vacation travel has always been a very important aspect of the modern economy, it’s been a way of life that almost everyone would inevitably experience, with the new platform of Airbnb coming into this field we see the touring business has been much more active nowadays. 
### Motivation
Setting prices for Airbnb listings becomes more and more challenging due to constantly changing demand and supply dynamics affected by various factors, which triggers the motivation behind this project, which is to provide property owners and travelers with more accurate and reliable information about Airbnb prices.
### Project Structure
* Dataset: listings.csv
* To Run the Program: main.py
* Visualize the Graph: the Figure Folder
### Dependency
* main.py: To excute the program
    * LoadDataset.py: To load the dataset
        * listings.csv
    * DataProcessor.py: To process the data and utilize PCA
    * FeatureExtraction.py: To preprocess the text data and utilize Word2Vec 
    * Visualize.py: To save the figure into Figure folder
    * Modeling.py: Defining ML models and a Evaluation helper function
### Example of Program Output
![Alt text](/Figure/Example_Output.png?raw=true "Example of Program Output")
### Note
Since the Word2Vec is used in this project, the performance result would be different in each run. But the best model and its performance will be printed in the last.