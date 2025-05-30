# Section 1: Project Definition

## Project Overview
The project aims to enhance Bertelsmann's understanding of its customer base through a comprehensive data analysis framework. It is divided into three key components:
1. **Data Exploration:** This initial phase focuses on identifying and addressing data quality issues within the provided demographic data. By cleaning and refining the dataset, the project ensures that subsequent analyses are based on accurate and reliable information.
2. **Customer Segmentation with Unsupervised Model:** An unsupervised machine learning model is employed to categorize individuals into distinct segments based on their behaviors and characteristics, allowing the company to uncover distinct customer traits and behaviors.
3. **Customer Prediction with Supervised Model:** A supervised machine learning model is utilized to predict future customers, enabling Bertelsmann to effectively target future mailout campaigns based on demographic data.

Overall, the project seeks to leverage data-driven insights to support more effective marketing and customer relationship strategies.

## Problem Statement
The primary problem to solve is to identify and predict potential customers for future mailout campaigns and to better understand the traits and behaviors of existing customers.

## Metrics
### Customer Segmentation
For customer segmentation, silhouette scores are used to measure the quality of clusters. Silhouette scores measure how similar a point is to its own cluster compared to other clusters, balancing both cohesion and separation. A high average silhouette score indicates well-defined, distinct clusters, making it a reliable diagnostic when ground truth labels are unavailable.

### Customer Prediction
For customer prediction, ROC AUC are used as the evaluation metric, because it seems to be the best fit for the business problem:
- offers a comprehensive evaluation of model performance by assessing the trade-off between recall and false positive rate across different thresholds
- maximizing recall (true positive rate) is essential to identify as many potential customers as possible
- control false positive rate is also crucial to avoid spamming uninterested users and protect brand reputation


# Section 2: Analysis

## Data Exploration
The initial phase involved exploring the demographic population datasets to identify data quality issues. Key findings include:

- **Data Types:** The datasets contained numeric and string types. String types such as `cameo_deu_2015` or `ost_west_kz` should be treated as categorical variables. Moreover, some numeric attributes are not ordinal (no perceived order) and should be handled as a categorical feature too, e.g. `finanztyp` or `shopper_typ`.


- **Unknown & Invalid Values:** Unknown and invalid values were identified based on the provided meta information and should be treated as missing data.


- **Missing Data (Record-Level):** Around 12% of all data records have more than 120 missing attributes (so over one third of the features), which should be removed due to the lack of information.


- **Missing Data (Feature-Level):** 14 attributes have a missing value ratio over 20% and should be completely dropped, because an uncertain imputation for that amount of data will possibly create too much bias. Some features like `ager_type` or `titel_kz` have a high missing ratio due to its logical nature and can be imputed purposeful. The remaining attributes have a max missing ratio of 7.5%, which can be kept in the dataset and impute the null values with its median for numeric types and mode for strings.


- **Feature Extraction:** `praegende_jugendjahre` (formative youth years) and `cameo_intl_2015` (social typology) are both compositions of multiple information and should be separated into multiple features.


- **Feature Correlation:** Various highly correlated features were identified such as `lp_status_fein` and `lp_status_grob`. Redundant features should be dropped to reduce dimensionality.


- **General Population vs. Customer Dataset:**
  - The customer dataset have 3 additional columns `customer_group`, `online_purchase` and `product_group`, which should be dropped to align the datasets.
  - There is one value in `gebaeudetyp` (categorical feature), that is only in the population dataset and therefore creates the corresponding dummy column only in the population dataset.
  - The share of records with more than one third of missing data is considerably higher in the customer dataset: 26.8% vs. 11.9%
  - The share of records with nearly complete data instead is higher in the customer dataset, e.g. less than 10 missing features: 59.9% vs. 48.6%
  - The customer data has a slightly lower share of features that contain more than 20% missing values: 2.8% vs. 5.5%
  - The missing ratios per feature between both datasets are mostly quite close, because the correlation is with 0.94 very high.

## Data Visualization
Several visualizations were created to better understand the datasets:

- **Missing Data (Record-Level):** The distribution of missing values per record in both population and customer datasets is helpful to observe the amount of missing data on the record-level and how the deviate compared to the other dataset.
  <br><br>
  Population Dataset:
  ![missing_values_count_per_record_population.png](img/missing_values_count_per_record_population.png)
  <br><br>
  Customer Dataset:
  ![missing_values_count_per_record_customer.png](img/missing_values_count_per_record_customer.png)


- **Missing Data (Feature-Level):** The missing values ratio per feature compared between both datasets shows, that they are quite similar for most of the features.
  <br><br>
  ![missing_values_ratio_per_feature.png](img/missing_values_ratio_per_feature.png)


- **Feature Correlation:** The correlation matrix shows clearly the high correlations between some redundant features.
  <br><br>
  ![correlation_matrix.png](img/correlation_matrix.png)


# Section 3: Methodology

## Data Preprocessing
Data preprocessing was a crucial step to ensure data quality and reliability for subsequent analyses. The revealed data issues from the initial data exploration were handled properly. The preprocessing involved several tasks:

- **Rectify Meta Attributes:** Attribute names in the metadata were aligned to match them with the demographic datasets.


- **Convert Unknown & Invalid Values:** Values that are out of range or marked as unknown in the meta information were converted to actual null values.


- **Remove Records / Features:** Records and features with too many missing values (records: > 33%, features: > 20%) were removed, due to the lack of information. 


- **Impute Missing Values:** The remaining missing data were imputed using median for numeric features and mode for categorical features.


- **Feature Engineering:** New features were extracted from existing ones (formative youth years and international social classifications)


- **Remove Uncertain Features:** Some features in the datasets have no provided meta information and could not be classified by their name or their values. Therefore, it is unclear weather to handle them as numeric or categorical feature.


- **Remove Redundant Features:** Highly correlated features were removed to reduce dimensionality.


- **Encoding:** All features were screened and classified as numeric or categorical. One-Hot-Encoding were applied to categorical features. Binary features were also standardized to 0/1.


- **Align Features:** Small deviations in the missing values between the population and customer dataset leads to different feature removals. Moreover, some categorical values are only in one of the datasets. Therefore, we have to ensure the same features in the same order for both datasets.


- **Scaling:** Numeric features will be standardized using the StandardScaler.


## Implementation

### Customer Segmentation

#### K-Means Clustering

K-Means were used for clustering, which partitions data into a predefined number of clusters by minimizing the distance between data points and their respective cluster centroids. 
It iteratively updates the centroids and reassigns points until the clusters stabilize. 
K-Means is a good choice for the customer segmentation because it is simple, efficient on large datasets, and often produces well-separated, interpretable clusters.

#### Optimal Number of Clusters

Since K-Means requires a predefined number of clusters, it is important to determine the optimal number for the dataset.
Common techniques for this are the elbow method and the silhouette score. To apply these techniques, different numbers of clusters were tested first and collect the corresponding results.

For the elbow method we have to plot the within-cluster sum of squares (WCSS or Inertia) against the number of clusters and identifying the "elbow point".
Unfortunately, the elbow curve did not provide a definitive indication of the optimal number of clusters, suggesting potential issues with cluster separation.

![elbow_curve_first.png](img/elbow_curve_first.png)

The silhouette score also was notably low with 0.05 as the highest value for 2 clusters, indicating poor cluster cohesion and separation.

![silhouette_score_first.png](img/silhouette_score_first.png)

#### Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while preserving as much variance as possible.

PCA is essential for K-Means clustering, because it helps eliminate noise and irrelevant features that can mislead the clustering process. This is why PCA can improve the accuracy and stability of K-Means clustering results.

The explained variance in PCA refers to the amount of the total variance in the data that is captured by each principal component. It indicates how much information from the original data is preserved by that component.

![cumulative_explained_variance_first.png](img/cumulative_explained_variance_first.png)

After PCA was applied while retaining 90% of the explained variance (resulting in 156 principal components), the silhouette score improved just marginally to 0.07. 

Therefore, further investigation was necessary, which are covered in the refinement section below.


### Customer Prediction

#### Model Selection

The goal was to predict weather or not an individual will be a potential customer. This means that it is a binary classification task that needs to be solved.

A possible algorithm approach are gradient boosting decision trees. They are highly effective and very successful in solving various machine learning problems.

An important aspect is the extreme **class imbalance** in the mailout dataset, that needs to be considered for the model development. Only about 1% of the data are customers.

One good choice could be XGBoost, that can appropriately handle class imbalance with the `scale_pos_weigth` parameter.
Furthermore, XGBoost offers an early stopping method to stop the training when the target metric do not increase anymore, which makes the training and tuning process more efficient.

Another popular gradient boosting framework is LightGBM. It also provides early stopping as well as a built-in handling for imbalanced classes. 
One option is to use the `scale_pos_weight` parameter like with XGBoost. Another possibility is the `is_unbalanced` parameter.

#### Split Training Data

To avoid data leakage and ensure an unbiased evaluation, a validation set for early stopping and hyperparameter tuning and a separate test set (data that the model has never seen before) for the final evaluation were used:
  - training set: for model training
  - validation set: for early stopping and hyperparameter tuning
  - test set: for final evaluation

Due to the high class imbalance, it is essential to use a stratified sampling approach to make sure, that the rare customers in the data are equally distributed.


## Refinement

### Customer Segmentation
Reducing PCA explained variance to 0.3 resulted in a better silhouette score of 0.16, although the clustering quality remains suboptimal.

The additional removal of the `d19` and `kba` features significantly reduced the dataset's dimensionality prior to applying PCA.
That approach in combination with an explained variance of 0.5 (resulted in 7 principal components) delivered the best silhouette score results. This will be discussed further in the model evaluation in section 4.


### Customer Prediction

#### Hyperparameter Tuning

Hyperparameter tuning is essential for achieving the best possible model performance.
One efficient way to tune the hyperparameters of the models is by using the Optuna framework. 
Optuna mainly uses a method called Tree-structured Parzen Estimator (TPE) as its optimization strategy, which is a type of Bayesian optimization:
- instead of trying random hyperparameters, Optuna models the probability distribution of good vs. bad hyperparameters
- it learns over time which areas of the search space are promising and then focusing more on these regions

The tuning process for the customer prediction models were controlled by setting:
- the ranges for each hyperparameter to be considered, e.g. the learning rate should be between 0.01 and 0.2
- the metric and its direction for optimization (maximize ROC AUC)
- the number of trials (100 trials were executed)

These tuning scales were used for the XGBoost model:

| Hyperparameter       | Range               | Log Scale |
|----------------------|---------------------|-----------|
| eta                  | 0.01 to 0.2         | Yes       |
| max_depth            | 3 to 10             | No        |
| min_child_weight     | 0.01 to 10          | Yes       |
| subsample            | 0.5 to 1.0          | No        |
| colsample_bytree     | 0.5 to 1.0          | No        |
| gamma                | 0.0001 to 10        | Yes       |
| lambda               | 0.0001 to 10        | Yes       |
| alpha                | 0.0001 to 10        | Yes       |


And for LightGBM:

| Hyperparameter       | Range               | Log Scale |
|----------------------|---------------------|-----------|
| learning_rate        | 0.001 to 0.02       | Yes       |
| num_leaves           | 20 to 300           | No        |
| max_depth            | -1 to 10            | No        |
| min_child_weight     | 0.01 to 10          | Yes       |
| lambda_l2            | 0.0001 to 10        | Yes       |
| lambda_l1            | 0.0001 to 10        | Yes       |
| feature_fraction      | 0.5 to 1.0          | No        |
| bagging_fraction      | 0.5 to 1.0          | No        |

#### Class Imbalance

A typical recommended value for the `scale_pos_weight` parameter is `sum(negative instances) / sum(positive instances)`.
Different adjustments for this weight were tested, but for the XGBoost model this was the best setting.

For the LightGBM model neither the `is_unbalanced` nor the `scale_pos_weight` parameter leads to suitable results.
Instead, we used the SMOTE library at the end to upsample the minority class in the training data, which works much better.

#### Target Metric Confusion

During the initial training sessions, the focus was on optimizing the AUCPR metric. However, achieving acceptable values proved to be nearly impossible. 

After some reflection, it became clear that this approach made little sense and that AUCPR simply wasn’t the right metric for this case. 
The response rate for email campaigns is inherently very low, which means concentrating on the precision metric was not particularly useful. It is low anyway.

Instead, it’s much more effective to focus on the recall (how many potential customers we are reaching) while also ensuring that the false-positive rate remains low, meaning we should minimize the number of emails that end up with uninterested customers.


# Section 4: Results

## Model Evaluation and Validation

### Customer Segmentation
After the refinement of the explained variance and further dimensionality reduction, we achieved a silhouette score of over 0.21 after all. 
This is still quite low, but could be sufficient for a basic customer segmentation.

![silhouette_score_best.png](img/silhouette_score_best.png)

We reached the highest scores for 2 and 8 clusters. For the customer segmentation the more granular clusters were taken, where a customer-heavy cluster was identified. 
This cluster (number 4) stands out with a proportion of 37% and is an appropriate base to gain insights about the customer characteristics based on that cluster.

![customer_proportions_per_cluster.png](img/customer_proportions_per_cluster.png)


#### Cluster Analysis I: Principal Component Loadings

The cluster traits were examined by its principal components to get an idea of how the cluster is positioned along each principal component.
Component #1 was clearly above the average, while component #3 was noticeably below.

| Principal Component | Value   |
|---------------------|---------|
| Component #1        | 3.29    |
| Component #2        | -0.30   |
| Component #3        | -1.99   |
| Component #4        | -0.01   |
| Component #5        | -0.26   |
| Component #6        | -0.16   |
| Component #7        | 0.13    |

Each principal component is a linear combination of the original variables. The loadings indicate how much each original variable contributes to each principal component. The sign of the loading (positive or negative) indicates the direction of the relationship.

The loadings of component #1 indicates relationships to these features:
- `lp_status_fein` - higher social status
- `finanz_minimalist` - higher financial interest
- `mobi_regio` - lower mobility
- `plz8_antg1` - higher share of 1-2 family houses
- `plz8_antg2` - less share of 3-5 family houses
- `konsumnaehe` - higher distance from building to Point of Sale
- `innenstadt` - higher distance to city center
- `zabeotyp` - consume more smart and green energy

The `semio` features dominate component #3, which describes different personal affinities:
- `semio_kaem` - less fightful attitude
- `semio_dom` - less dominant minded
- `semio_krit` - less critical minded
- `semio_erl` - less eventful orientated
- `semio_rat` - less rational minded
- `semio_kult` - more cultural minded
- `semio_soz` - more social minded
- `semio_fam` - more familiar minded
- `semio_vert` - more dreamily


#### Cluster Analysis II: Original Data Values

The original values of the customer-heavy cluster were also compared with the total population data to discover specific traits of that cluster.
To interpret the values correctly, the original data was cleaned, but not scaled or encoded.

**Numeric features** were evaluated by comparing the means for the customer-heavy cluster and total dataset.

Some of the largest differences (relative deviation >30%) that were observed fits well to the findings of the analyzed principal components, but there are also some additional characteristics revealed:

| Feature             | Compared to Average Population              | Cluster Average Value       |
|---------------------|---------------------------------------------|-----------------------------|
| anz_haushalte_aktiv | less households per building                | 1-2 households per building |
| anz_kinder          | less children                               | 0.022                       |
| anz_titel           | more individuals with academic title        | no academic title           |
| cameo_deug_2015     | higher social status                        | established middleclass     |
| finanz_anleger      | more financial investors                    | high to very high           |
| finanz_minimalist   | higher financial interest                   | low interest = very low     |
| finanz_hausbauer    | higher financial main focus on an own house | high focus                  |
| hh_einkommen_score  | higher income                               | high income                 |
| lp_status_fein      | higher social status                        | (new) houseowners           |
| mobi_regio          | lower mobility                              | low mobility                |
| plz8_antg3          | less 6-10 family houses                     | low share                   |
| plz8_antg4          | less >10 family houses                      | none                        |
| plz8_baumax         | more 1-2 family houses                      | mainly 1-2 family homes     |
| semio_kaem          | less >10 family houses                      | none                        |
| semio_vert          | lower dreamily affinity                     | very low affinity           |
| zabeotyp            | smarter energy consumption                  | smart                       |

An interesting observation is, that we saw an inverse relationship between a dreamy affinity and the principal component #3 before, but the analysis of the original values cleary show that that assumption was misleading.

To assess the **categorical variables**, each categorical value was counted in the cluster as well as in the total dataset.

The Top20 values with the highest share compared to the average population reveals some interesting customer traits:
- `titel_kz` - higher share of academic titles
- `cameo_deu_2015` - most of the upper class and upper middle class typologies are represented in the Top20 values with the highest share
- `anrede_kz` - more male customers
- `ager_typ` - more best-ager customers
- `gebaeudetyp` - higher share of buildings without actually known households
- `gfk_urlaubertyp` - more hiker and golden ager regarding the vacation habits
- `finanztyp` - more money savers as well as investors
- `shopping_type` - more individuals who seek pleasure and enjoyment from their shopping experiences
- `alterskategorie_grob` - higher share of >60 years old

Some of them were already uncovered in the previous analyses (higher social class, investors), but emphasized these insights.

### Customer Prediction

#### Best Hyperparameters
These hyperparameter settings achieved the highest AUC score for the models:

| XGBoost Hyperparameter | Value                            |
|------------------------|----------------------------------|
| eta                    | 0.0345                           |
| max_depth              | 4                                |
| min_child_weight       | 0.9146                           |
| subsample              | 0.596                            |
| colsample_bytree       | 0.7637                           |
| gamma                  | 0.1327                           |
| lambda                 | 0.4347                           |
| alpha                  | 0.0012                           |

In the XGB model, the learning rate (`eta`) of 0.03 allows for a gradual learning process.
The `max depth` of 4 indicates a relatively deeper tree, allowing the model to capture more complex interactions in the data.
A `min_child_weight` of 0.91 suggests that the model is less sensitive to overfitting, as it requires fewer instances in a leaf node before making a split.
The regularization parameters, with `lambda` at 0.43 and `alpha` at 0.001, imply a moderate level of regularization to control overfitting while still allowing the model to learn important features from the data.

| LightGBM Hyperparameter | Value                            |
|-------------------------|----------------------------------|
| learning_rate           | 0.0189                           |
| num_leaves              | 134                              |
| max_depth               | 1                                |
| min_child_weight        | 3.516                            |
| lambda_l2               | 3.792                            |
| lambda_l1               | 0.0002                           |
| feature_fraction        | 0.5507                           |
| bagging_fraction        | 0.7085                           |

The LGBM model has a `learning rate` of 0.02, indicating a moderate pace of learning, which helps in achieving a balance between convergence speed and overfitting.
The model uses 134 leaves (`num leaves`), suggesting a complex tree structure that can capture intricate patterns in the data.
With a `max depth` of 1, this indicates a shallow tree, which may be a result of using a high number of leaves to prevent overfitting while still allowing for flexibility in learning.
The `min_child_weight` of 3.5 suggests that the model requires a minimum sum of instance weight in a child node to prevent overfitting, 
while the regularization parameters (`lambda_l2` and `lambda_l1`) help control complexity.

#### Training Curves
Although both frameworks using early stopping, it is crucial to assess overfitting and underfitting by using training curves, 
because it helps identify whether a model is learning the underlying patterns in the data or simply memorizing the training samples.

The XGBoost model seems to be well-tuned, with no signs of significant overfitting or underfitting based on the log loss curves:

![logloss_curve_xgb.png](img/logloss_curve_xgb.png)

The number of trees appears also valid with the highest AUC score for the validation set at the end:
![auc_curve_xgb.png](img/auc_curve_xgb.png)

The LightGBM model struggles first with class imbalance while using the built-in parameters to handle it. The result was a steadily increasing logloss curve:

![binary_logloss_curve_lgb_first.png](img/binary_logloss_curve_lgb_first.png)

With the use of SMOTE the model performance improves considerably and the loss curve looks appropriate without a noticeable over- or underfitting:

![binary_logloss_curve_lgb_best.png](img/binary_logloss_curve_lgb_best.png)


## Justification

### Customer Segmentation

It is generally better to analyze a cluster by the original data values to get clear, actionable insights.
Principal component loadings are harder to explain, but it's still useful to check them because they show which features together shape the cluster.

The revealed customer profile appears coherent, with no obvious contradictions, which is a good sign that the clustering results from the model are reliable.

This could be the summerized cluster profile that describes the characteristics of the customers:

- Demographic Characteristics:
  - Social Status: The cluster is predominantly composed of individuals with a higher social status, categorized as established middle class and upper class. A significant share of the population holds academic titles, indicating a well-educated demographic.
  - Age Distribution: There is a notable presence of "best-ager" customers, with a higher concentration of individuals aged 60 and above.
  - Gender Composition: The cluster has a higher proportion of male customers.
<br><br>
- Housing and Living Situation:
  - Residential Characteristics: The cluster is characterized by a preference for 1-2 family houses, with a low share of multi-family housing. Many residents live in areas with fewer households per building, indicating a more spacious living environment.
  - Mobility: Residents exhibit lower mobility, with a higher distance from Points of Sale and city centers, suggesting a more suburban lifestyle.
<br><br>
- Financial Behavior:
  - Financial Interests: The cluster shows a strong inclination towards financial investments, with many individuals categorized as financial investors and money savers. There is a significant focus on home ownership, reflecting a preference for investing in personal property.
  - Income Levels: The average income within this cluster is higher than the general population, indicating a financially stable group.
<br><br>
- Consumption Patterns:
  - Energy Consumption: Customers in this segment are inclined towards smart and green energy solutions, reflecting a commitment to sustainability.
  - Shopping Preferences: The cluster members derive pleasure and enjoyment from shopping experiences, indicating a preference for leisure-oriented purchasing behavior.
<br><br>
- Personal Affinities:
  - Mindset and Attitudes: The cluster is characterized by less dominant, critical and rational thinking. Instead, members display a more cultural, social and familial mindset with a tendency towards a less eventful orientation.


### Customer Prediction

The ROC curves provide several insights about the performance of the models:
- the curve plots the true positive rate (TPR) against the false positive rate (FPR)
  - a higher TPR indicates that the model is effectively identifying the customers (positive cases)
  - a lower FPR indicates that it is not incorrectly labeling too many negative cases as positive (sending mails to uninterested people)
- both models perform better than random guessing and have a good ability to distinguish between customers and non-customers
- however, XGBoost shows a stronger ability with an AUC of 0.795 compared to LightGBM with 0.767
- the ROC curve rises steeply in the lower left corner, which is favorable, as it indicates that the models achieve a high true positive rate (identify many potential customers) with a low false positive rate (sending unnecessary mails) at lower thresholds
- the relatively gradual slope in the mid-range suggests that the "spam rate" significantly increases while only a limited number of additional customers are being identified

A possibly good trade-off for the XGBoost model would be at around 0.86 TPR and 0.30 FPR, which means that 86% of the potential customers were identified while avoiding 70% of unnecessary "spam" mails.

![roc_curves.png](img/roc_curves.png)

The threshold on the predicted probabilities for the suggestest trade-off above would be around `0.37`.
This means that any predicted probability value of the model over the threshold value will be classified as a potential customer.

![tpr_fpr_thresholds_xgb.png](img/tpr_fpr_thresholds_xgb.png)


# Section 5: Conclusion

## Reflection

### Problem Solution
**Data Exploration:**
The project begins with a thorough exploration of the demographic datasets, identifying and rectifying data quality issues such as unknown values, missing data and irrelevant features. This phase ensures that the dataset is clean and reliable for further analysis.

**Customer Segmentation:** 
An unsupervised machine learning model (K-Means) is employed to classify customers into distinct segments based on their behaviors and characteristics. The optimal number of clusters is determined using techniques like the elbow method and silhouette scores, although challenges arise due to low cohesion and separation in the clusters.

**Customer Prediction:** 
Supervised learning models (XGBoost and LightGBM) are utilized to predict potential customers for future campaigns. The models are trained on a stratified dataset to handle the extreme class imbalance present in the data. Hyperparameter tuning is performed to optimize model performance, focusing on maximizing the ROC AUC metric.

### Challenges

**Data Quality Challenges**:
One of the most challenging aspects of the project was addressing the data quality issues during the exploration phase. 
With a significant portion of the data containing missing values and unknown entries, ensuring that the dataset was 
clean and suitable for analysis required careful handling. This step was crucial for the integrity of the subsequent analyses.

**Cluster Cohesion and Separation:** 
Achieving meaningful customer segments through K-Means clustering proved difficult. The low silhouette scores indicated 
that the clusters were not well-defined, prompting further refinement through dimensionality reduction techniques like PCA.
This aspect highlighted the complexities involved in unsupervised learning, particularly in finding a balance between 
reducing dimensionality and retaining valuable information for effective segmentation.

**Class Imbalance Challenges:** A major challenge in the prediction phase was the class imbalance in the dataset, 
with only about 1% of the data representing actual customers. This imbalance risked biasing the models towards the majority class of non-customers, 
making it difficult to accurately identify potential customers. 
To address this, techniques like SMOTE for oversampling the minority class and adjusting class weights were necessary, 


## Improvement

Some suggestions for future research to improve the solution could be:
- Implement more extensive hyperparameter optimization strategies to further fine-tune model parameters
- Using SHAP to better understand the influence of various features on predictions
- Explore more sophisticated sampling methods to handle the class imbalance
- Test other model types like Random Forest or SVMs for the customer prediction and DBSCAN or Hierarchical Clustering for the customer segmentation