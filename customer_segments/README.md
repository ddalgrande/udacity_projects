# Identifying Customer Segments with Arvato

## Project Overview

In this project, our Bertelsmann partners AZ Direct and Arvato Financial Solutions have provided two datasets one with demographic information about the people of Germany, and one with that same information for customers of a mail-order sales company. You’ll look at relationships between demographics features, organize the population into clusters, and see how prevalent customers are in each of the segments obtained.

## Pre-requisites 

This project uses Python 3 and is designed to be completed through the Jupyter Notebooks IDE. It is highly recommended that you use the Anaconda distribution to install Python, since the distribution includes all necessary Python libraries as well as Jupyter Notebooks. The following libraries are expected to be used in this project:

- NumPy
- pandas
- Sklearn / scikit-learn
- Matplotlib (for data visualization)
- Seaborn (for data visualization)

## Project Details

Below, you will find an outline of the steps that you will take in this project. It is recommended that you read through all of these instructions before starting on the project so you can build a high-level understanding of the tasks ahead!

### Step 0: The Data

There are a number of files used for this project available in the repository. The files available for the project are:

- `Udacity_AZDIAS_Subset.csv`: Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- `Udacity_CUSTOMERS_Subset.csv`: Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- `Data_Dictionary.md`: Information file about the features in the provided datasets.
- `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographic data.
- `Identify_Customer_Segments.ipynb`: Jupyter Notebook divided into sections and guidelines for completing the project. The notebook provides more details and tips than the outline given here.

After downloading the data and unzipping it to a working folder of your choice, use the Terminal or Command Prompt to navigate to your selected folder and execute the command `jupyter notebook Identify_Customer_Segments.ipynb` to open up a browser window or tab to work on the project notebook.

### Step 1: Preprocessing

When you start an analysis, you must first explore and understand the data that you are working with. In this (and the next) step of the project, you’ll be working with the general demographics data. As part of your investigation of dataset properties, you must attend to a few key points:

- How are missing or unknown values encoded in the data? Are there certain features (columns) that should be removed from the analysis because of missing data? Are there certain data points (rows) that should be treated separately from the rest?
- Consider the level of measurement for each feature in the dataset (e.g. categorical, ordinal, numeric).
- What assumptions must be made in order to use each feature in the final analysis? Are there features that need to be re-encoded before they can be used? Are there additional features that can be dropped at this stage?

You will create a cleaning procedure that you will apply first to the general demographic data, then later to the customers data.

### Step 2: Feature Transformation

Now that your data is clean, you will use dimensionality reduction techniques to identify relationships between variables in the dataset, resulting in the creation of a new set of variables that account for those correlations. In this stage of the project, you will attend to the following points:

- The first technique that you should perform on your data is feature scaling. What might happen if we don’t perform feature scaling before applying later techniques you’ll be using?
- Once you’ve scaled your features, you can then apply principal component analysis (PCA) to find the vectors of maximal variability. How much variability in the data does each principal component capture? 
- Can you interpret associations between original features in your dataset based on the weights given on the strongest components? How many components will you keep as part of the dimensionality reduction process?

You will use the **sklearn** library to create objects that implement your feature scaling and PCA dimensionality reduction decisions.

### Step 3: Clustering

Finally, on your transformed data, you will apply clustering techniques to identify groups in the general demographic data. You will then apply the same clustering model to the customers dataset to see how market segments differ between the general population and the mail-order sales company. You will tackle the following points in this stage:

- Use the k-means method to cluster the demographic data into groups. How should you make a decision on how many clusters to use?
- Apply the techniques and models that you fit on the demographic data to the customers data: data cleaning, feature scaling, PCA, and k-means clustering. Compare the distribution of people by cluster for the customer data to that of the general population. Can you say anything about which types of people are likely consumers for the mail-order sales company?