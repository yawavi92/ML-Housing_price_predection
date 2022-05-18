# Housing Price Prediction
[Melbourne, Australia Housing Price Prediction](https://melbourne-housing.herokuapp.com/)

![Houses Picture](static/images/melbourne_houses.jpg)

## Table of Contents: :house_with_garden:
01. [Reason For Our Project](#reason)
02. [Data Source](#source)
03. [Technologies](#tech)
04. [Approach](#approach)
05. [Flowchart](#flowchart)
06. [Extract, Transform, Load](#etl)
07. [Exploratory Data Analysis](#eda)
08. [Regression Model Results](#models)
09. [Best model](#model)
10. [The Website](#site)
11. [Going Forward](#forward)
12. [Contributors](#team)

<a name="reason"></a>
## Reason For Our Project: :thinking:
For the average consumer, finding appropriate housing within a set budget can be a daunting task. Predicting housing prices are expected to help people who plan to buy a house so they can know the price range in the future and be able to plan their finances. Housing price predictions are also beneficial for property investors to know the trend of housing prices in a certain location.

<a name="source"></a>
## Data Source: :books:
This is a house prices dataset for Melbourne, Australia from September 2017 to September 2019. This data was scraped from publicly available results posted every week from Domain.com.au. The dataset includes address, type of real estate, suburb, method of selling, rooms, price, real estate agent, date of sale, property size, land size, council area and distance from the Central Business District. From the dataset provided, we chose to clean our data to focus on the important features to be used within our machine learning models to determine which would give us the best predicted housing price.

- [Kaggle: Melbourne Housing Market](https://www.kaggle.com/anthonypino/melbourne-housing-market)

<a name="tech"></a>
## Technologies: :building_construction::hammer_and_wrench:
- Jupyter Notebook
- Python
- Numpy
- Pandas
- Seaborn
- Matplotlib
- SciPy
- PostgreSQL/ SQLAlchemy
- Scikit-learn
- Flask
- HTML/CSS
- Heroku

<a name="approach"></a>
## Approach: :computer:
01. Identify data source
02. Collect and clean Melbourne, Australia housing data
03. Normalized target feature for better model results
04. Create charts and graphs using Pyplot
05. Load data into PostgreSQL
06. Transform data to be fitted into models
07. Test varying machine learning models and determine best option
08. Customize HTML and CSS for final application
09. Develope Flask application for model deployment
10. Visualize dashboard in Heroku

<a name="flowchart"></a>
## Flowchart: :memo:

![Flowchart](static/images/architectural_diagram.png)

<a name="etl"></a>
## Extract, Transform, Load: :magic_wand:
The data was provided to us in a CSV. We checked to verify the datatypes, dropped any columns we didn't need and renamed some of the columns we kept. We also checked for any null values in the dataset and dropped them. Once our data checked out, we made sure not to have any duplicate data values. From there, we plot a heatmap to determine which features are important to our target feature and drop those we don't need.

![ETL01](static/images/etl_01.png)
![ETL02](static/images/etl_02.png)
![Heatmap](static/images/features_heatmap.jpg)

<a name="eda"></a>
## Exploratory Data Analysis: :mag::bar_chart:
After being satisfied with our cleaned data, we proceeded to our data exploration. We used Matplotlib and Seaborn to do visualizations and visualized many different aspects of our data to see how it reads.

![Normalization](static/images/price_normalizing.png)

We normalized our target feature to get a better outcome on our predictions.

![Price_Bathroom](static/images/price_bathroom_relationship.jpg)
![Price_Bedroom](static/images/price_bedroom_relationship.jpg)
![Price_Bedroom_Bathroom](static/images/price_bedroom_bathroom_relationship.jpg)

We checked the relationship of price with bedrooms and bathrooms.

<a name="regression"></a>
## Regression Models Results: :nerd_face:

![Linear_Regression](static/images/lr_model.png)
![Random_Forest](static/images/rfr_model.png)
![Decision_Tree](static/images/dtr_model.png)
![Randomized_Search_CV](static/images/rscv_model.png)
![Support_Vector_Regressor](static/images/svr_model.png)
![Lasso](static/images/lasso_model.png)
![Ridge](static/images/ridge_model.png)

<a name="model"></a>
## Best model: :trophy:
![Decision Tree](static/images/Decision_Tree_Model.png)


## The Website: :electron:
![Website](static/images/melbourne_housing_website.png)

<a name="forward"></a>
## Going Forward: :bulb:
- Visualize data to show housing price trends on a map through an application such as Tableau
- ​Scrape more recent data and see if the trends and predictions hold true
- Update the look and feel of the website to make it more user friendly
- Redefine the parameters of our models to try and fix possible overfitting

<a name="team"></a>
## Contributors: :brain:

- [Marius Clark](https://github.com/Zone6Mars)
- [Darryl Connelly II](https://github.com/DConnellyII)
- [Yawavi Koudjonou](https://github.com/yawavi92)
- [Kyungchul Oh](https://github.com/KCOH)
