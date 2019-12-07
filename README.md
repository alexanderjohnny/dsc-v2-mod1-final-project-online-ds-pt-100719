
# Final Project Submission

Please fill out:
* Student name: alex beat
* Student pace: self paced / part time / full time: part time
* Scheduled project review date/time: na
* Instructor name: James Irving
* Blog post URL: https://medium.com/@jonathonalexander/software-engineer-or-data-scientist-3493360685d8
* Video of 5-min Non-Technical Presentation: na


# TABLE OF CONTENTS 

*Click to jump to matching Markdown Header.*<br><br>

<font size=4rem>
    
- [Introduction](#INTRODUCTION)<br>
- **[OBTAIN](#OBTAIN)**<br>
- **[SCRUB](#SCRUB)**<br>
- **[EXPLORE](#EXPLORE)**<br>
- **[MODEL](#MODEL)**<br>
- **[iNTERPRET](#iNTERPRET)**<br>
- [Conclusions/Recommendations](#CONCLUSIONS-&-RECOMMENDATIONS)<br>
</font>
___


## PROCESS CHECKLIST


1. **[OBTAIN](#OBTAIN)**
    - Import data, inspect, check for datatypes to convert and null values
    - Display header and info.
    - Drop any unneeded columns, if known (`df.drop(['col1','col2'],axis=1,inplace=True`)
    <br><br>

2. **[SCRUB](#SCRUB)**
    - Recast data types, identify outliers, check for multicollinearity, normalize data**
    - Check and cast data types
        - [ ] Check for #'s that are store as objects (`df.info()`,`df.describe()`)
            - when converting to #'s, look for odd values (like many 0's), or strings that can't be converted.
            - Decide how to deal weird/null values (`df.unique()`, `df.isna().sum()`)
            - `df.fillna(subset=['col_with_nulls'],'fill_value')`, `df.replace()`
        - [ ] Check for categorical variables stored as integers.
            - May be easier to tell when you make a scatter plotm or `pd.plotting.scatter_matrix()`
            
    - [ ] Check for missing values  (df.isna().sum())
        - Can drop rows or colums
        - For missing numeric data with median or bin/convert to categorical
        - For missing categorical data: make NaN own category OR replace with most common category
    - [ ] Check for multicollinearity
        - Use seaborn to make correlation matrix plot 
        - Good rule of thumb is anything over 0.75 corr is high, remove the variable that has the most correl with the largest # of variables
    - [ ] Normalize data (may want to do after some exploring)
        - Most popular is Z-scoring (but won't fix skew) 
        - Can log-transform to fix skewed data
    
3. **[EXPLORE](#EXPLORE)**
    - [ ] Check distributions, outliers, etc**
    - [ ] Check scales, ranges (df.describe())
    - [ ] Check histograms to get an idea of distributions (df.hist()) and data transformations to perform.
        - Can also do kernel density estimates
    - [ ] Use scatter plots to check for linearity and possible categorical variables (`df.plot("x","y")`)
        - categoricals will look like vertical lines
    - [ ] Use `pd.plotting.scatter_matrix(df)` to visualize possible relationships
    - [ ] Check for linearity.
   
4. **[MODEL](#MODEL)**

    - **Fit an initial model:** 
        - Run an initial model and get results

    - **Holdout validation / Train/test split**
        - use sklearn `train_test_split`
    
5. **[iNTERPRET](#iNTERPRET)**
    - **Assessing the model:**
        - Assess parameters (slope,intercept)
        - Check if the model explains the variation in the data (RMSE, F, R_square)
        - *Are the coeffs, slopes, intercepts in appropriate units?*
        - *Whats the impact of collinearity? Can we ignore?*
        <br><br>
    - **Revise the fitted model**
        - Multicollinearity is big issue for lin regression and cannot fully remove it
        - Use the predictive ability of model to test it (like R2 and RMSE)
        - Check for missed non-linearity
        
- **Interpret final model and draw >=3 conclusions and recommendations from dataset**

# INTRODUCTION

> Explain the point of your project and what question you are trying to answer with your modeling.

The purpose of this project is to clean, explore, and model this dataset with a multivariate linear regression to predict the sale price of houses as accurately as possible. 



# OBTAIN


```python
# OBTAIN

# Import data, inspect, check for datatypes to convert and null values
# Display header and info.
# Drop any unneeded columns, if known (df.drop(['col1','col2'],axis=1,inplace=True)
```


```python
!pip install -U fsds_100719
from fsds_100719.imports import *
```

    Requirement already up-to-date: fsds_100719 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (0.4.45)
    Requirement already satisfied, skipping upgrade: tzlocal in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (2.0.0)
    Requirement already satisfied, skipping upgrade: pprint in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.1)
    Requirement already satisfied, skipping upgrade: IPython in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (7.8.0)
    Requirement already satisfied, skipping upgrade: ipywidgets in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (7.5.1)
    Requirement already satisfied, skipping upgrade: missingno in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.4.2)
    Requirement already satisfied, skipping upgrade: scipy in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.3.1)
    Requirement already satisfied, skipping upgrade: scikit-learn in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.21.2)
    Requirement already satisfied, skipping upgrade: pandas in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.25.1)
    Requirement already satisfied, skipping upgrade: pandas-profiling in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (2.3.0)
    Requirement already satisfied, skipping upgrade: wordcloud in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.6.0)
    Requirement already satisfied, skipping upgrade: numpy in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.16.5)
    Requirement already satisfied, skipping upgrade: pyperclip in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.7.0)
    Requirement already satisfied, skipping upgrade: matplotlib in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (3.1.1)
    Requirement already satisfied, skipping upgrade: seaborn in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.9.0)
    Requirement already satisfied, skipping upgrade: pytz in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from tzlocal->fsds_100719) (2019.2)
    Requirement already satisfied, skipping upgrade: backcall in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.1.0)
    Requirement already satisfied, skipping upgrade: appnope; sys_platform == "darwin" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.1.0)
    Requirement already satisfied, skipping upgrade: traitlets>=4.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (4.3.2)
    Requirement already satisfied, skipping upgrade: decorator in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (4.4.0)
    Requirement already satisfied, skipping upgrade: jedi>=0.10 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.15.1)
    Requirement already satisfied, skipping upgrade: pexpect; sys_platform != "win32" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (4.7.0)
    Requirement already satisfied, skipping upgrade: pickleshare in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.7.5)
    Requirement already satisfied, skipping upgrade: setuptools>=18.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (41.2.0)
    Requirement already satisfied, skipping upgrade: pygments in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (2.4.2)
    Requirement already satisfied, skipping upgrade: prompt-toolkit<2.1.0,>=2.0.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (2.0.9)
    Requirement already satisfied, skipping upgrade: ipykernel>=4.5.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipywidgets->fsds_100719) (5.1.2)
    Requirement already satisfied, skipping upgrade: widgetsnbextension~=3.5.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipywidgets->fsds_100719) (3.5.1)
    Requirement already satisfied, skipping upgrade: nbformat>=4.2.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipywidgets->fsds_100719) (4.4.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from scikit-learn->fsds_100719) (0.13.2)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.6.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas->fsds_100719) (2.8.0)
    Requirement already satisfied, skipping upgrade: jinja2>=2.8 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (2.10.1)
    Requirement already satisfied, skipping upgrade: confuse>=1.0.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (1.0.0)
    Requirement already satisfied, skipping upgrade: htmlmin>=0.1.12 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (0.1.12)
    Requirement already satisfied, skipping upgrade: phik>=0.9.8 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (0.9.8)
    Requirement already satisfied, skipping upgrade: astropy in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (3.2.3)
    Requirement already satisfied, skipping upgrade: pillow in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from wordcloud->fsds_100719) (6.1.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from matplotlib->fsds_100719) (0.10.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from matplotlib->fsds_100719) (1.1.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from matplotlib->fsds_100719) (2.4.2)
    Requirement already satisfied, skipping upgrade: ipython_genutils in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from traitlets>=4.2->IPython->fsds_100719) (0.2.0)
    Requirement already satisfied, skipping upgrade: six in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from traitlets>=4.2->IPython->fsds_100719) (1.12.0)
    Requirement already satisfied, skipping upgrade: parso>=0.5.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jedi>=0.10->IPython->fsds_100719) (0.5.1)
    Requirement already satisfied, skipping upgrade: ptyprocess>=0.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pexpect; sys_platform != "win32"->IPython->fsds_100719) (0.6.0)
    Requirement already satisfied, skipping upgrade: wcwidth in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from prompt-toolkit<2.1.0,>=2.0.0->IPython->fsds_100719) (0.1.7)
    Requirement already satisfied, skipping upgrade: jupyter-client in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipykernel>=4.5.1->ipywidgets->fsds_100719) (5.3.3)
    Requirement already satisfied, skipping upgrade: tornado>=4.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipykernel>=4.5.1->ipywidgets->fsds_100719) (6.0.3)
    Requirement already satisfied, skipping upgrade: notebook>=4.4.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (5.7.8)
    Requirement already satisfied, skipping upgrade: jsonschema!=2.5.0,>=2.4 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbformat>=4.2.0->ipywidgets->fsds_100719) (3.0.2)
    Requirement already satisfied, skipping upgrade: jupyter_core in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbformat>=4.2.0->ipywidgets->fsds_100719) (4.5.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jinja2>=2.8->pandas-profiling->fsds_100719) (1.1.1)
    Requirement already satisfied, skipping upgrade: pyyaml in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from confuse>=1.0.0->pandas-profiling->fsds_100719) (5.1.2)
    Requirement already satisfied, skipping upgrade: numba>=0.38.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (0.46.0)
    Requirement already satisfied, skipping upgrade: pytest-pylint>=0.13.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (0.14.1)
    Requirement already satisfied, skipping upgrade: pytest>=4.0.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (5.3.0)
    Requirement already satisfied, skipping upgrade: nbconvert>=5.3.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (5.5.0)
    Requirement already satisfied, skipping upgrade: pyzmq>=13 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets->fsds_100719) (18.1.0)
    Requirement already satisfied, skipping upgrade: Send2Trash in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (1.5.0)
    Requirement already satisfied, skipping upgrade: prometheus-client in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (0.7.1)
    Requirement already satisfied, skipping upgrade: terminado>=0.8.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (0.8.2)
    Requirement already satisfied, skipping upgrade: pyrsistent>=0.14.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->fsds_100719) (0.14.11)
    Requirement already satisfied, skipping upgrade: attrs>=17.4.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->fsds_100719) (19.1.0)
    Requirement already satisfied, skipping upgrade: llvmlite>=0.30.0dev0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from numba>=0.38.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.30.0)
    Requirement already satisfied, skipping upgrade: pylint>=1.4.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (2.4.4)
    Requirement already satisfied, skipping upgrade: importlib-metadata>=0.12; python_version < "3.8" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (0.17)
    Requirement already satisfied, skipping upgrade: py>=1.5.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (1.8.0)
    Requirement already satisfied, skipping upgrade: packaging in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (19.2)
    Requirement already satisfied, skipping upgrade: more-itertools>=4.0.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (7.0.0)
    Requirement already satisfied, skipping upgrade: pluggy<1.0,>=0.12 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (0.12.0)
    Requirement already satisfied, skipping upgrade: bleach in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (1.5.0)
    Requirement already satisfied, skipping upgrade: defusedxml in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.6.0)
    Requirement already satisfied, skipping upgrade: testpath in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.4.2)
    Requirement already satisfied, skipping upgrade: pandocfilters>=1.4.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (1.4.2)
    Requirement already satisfied, skipping upgrade: mistune>=0.8.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.8.4)
    Requirement already satisfied, skipping upgrade: entrypoints>=0.2.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.3)
    Requirement already satisfied, skipping upgrade: mccabe<0.7,>=0.6 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (0.6.1)
    Requirement already satisfied, skipping upgrade: isort<5,>=4.2.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (4.3.21)
    Requirement already satisfied, skipping upgrade: astroid<2.4,>=2.3.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (2.3.3)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from importlib-metadata>=0.12; python_version < "3.8"->pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (0.5.1)
    Requirement already satisfied, skipping upgrade: html5lib!=0.9999,!=0.99999,<0.99999999,>=0.999 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from bleach->nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.9999999)
    Requirement already satisfied, skipping upgrade: typed-ast<1.5,>=1.4.0; implementation_name == "cpython" and python_version < "3.8" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from astroid<2.4,>=2.3.0->pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (1.4.0)
    Requirement already satisfied, skipping upgrade: wrapt==1.11.* in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from astroid<2.4,>=2.3.0->pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (1.11.2)
    Requirement already satisfied, skipping upgrade: lazy-object-proxy==1.4.* in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from astroid<2.4,>=2.3.0->pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (1.4.3)



```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
```

## Import data, inspect, display header and info.


```python
pd.set_option('display.max_columns',0)
```


```python
df = pd.read_csv('kc_house_data.csv')
display(df.head())
display(df.info())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21597 non-null int64
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB



    None


## Check for repeat rows


```python
#check for repeat rows
df['id'].value_counts()

```




    795000620     3
    1825069031    2
    2019200220    2
    7129304540    2
    1781500435    2
                 ..
    7812801125    1
    4364700875    1
    3021059276    1
    880000205     1
    1777500160    1
    Name: id, Length: 21420, dtype: int64




```python
# drop repeat rows based on 'id'
df.drop_duplicates(subset='id', inplace=True)
df['id'].value_counts()
```




    2911700010    1
    5450300010    1
    5104511600    1
    1160000115    1
    686530110     1
                 ..
    2115510470    1
    2922701305    1
    6071600370    1
    526059224     1
    1777500160    1
    Name: id, Length: 21420, dtype: int64



## Check for null vals


```python
# check for null values and how many
display(df.isna().any())
display(df.isna().sum())
```


    id               False
    date             False
    price            False
    bedrooms         False
    bathrooms        False
    sqft_living      False
    sqft_lot         False
    floors           False
    waterfront        True
    view              True
    condition        False
    grade            False
    sqft_above       False
    sqft_basement    False
    yr_built         False
    yr_renovated      True
    zipcode          False
    lat              False
    long             False
    sqft_living15    False
    sqft_lot15       False
    dtype: bool



    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2353
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3804
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64


### Fix waterfront nulls


```python
#waterfront with null value
df.loc[df['waterfront'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1736800520</td>
      <td>4/3/2015</td>
      <td>662500.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>3560</td>
      <td>9796</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1860</td>
      <td>1700.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98007</td>
      <td>47.6007</td>
      <td>-122.145</td>
      <td>2210</td>
      <td>8925</td>
    </tr>
    <tr>
      <td>23</td>
      <td>8091400200</td>
      <td>5/16/2014</td>
      <td>252700.0</td>
      <td>2</td>
      <td>1.50</td>
      <td>1070</td>
      <td>9643</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1070</td>
      <td>0.0</td>
      <td>1985</td>
      <td>NaN</td>
      <td>98030</td>
      <td>47.3533</td>
      <td>-122.166</td>
      <td>1220</td>
      <td>8386</td>
    </tr>
    <tr>
      <td>40</td>
      <td>5547700270</td>
      <td>7/15/2014</td>
      <td>625000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2570</td>
      <td>5520</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>2570</td>
      <td>0.0</td>
      <td>2000</td>
      <td>NaN</td>
      <td>98074</td>
      <td>47.6145</td>
      <td>-122.027</td>
      <td>2470</td>
      <td>5669</td>
    </tr>
    <tr>
      <td>55</td>
      <td>9822700295</td>
      <td>5/12/2014</td>
      <td>885000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2830</td>
      <td>5000</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>2830</td>
      <td>0.0</td>
      <td>1995</td>
      <td>0.0</td>
      <td>98105</td>
      <td>47.6597</td>
      <td>-122.290</td>
      <td>1950</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21578</td>
      <td>5087900040</td>
      <td>10/17/2014</td>
      <td>350000.0</td>
      <td>4</td>
      <td>2.75</td>
      <td>2500</td>
      <td>5995</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2500</td>
      <td>0.0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98042</td>
      <td>47.3749</td>
      <td>-122.107</td>
      <td>2530</td>
      <td>5988</td>
    </tr>
    <tr>
      <td>21582</td>
      <td>8956200760</td>
      <td>10/13/2014</td>
      <td>541800.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3118</td>
      <td>7866</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3</td>
      <td>9</td>
      <td>3118</td>
      <td>0.0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98001</td>
      <td>47.2931</td>
      <td>-122.264</td>
      <td>2673</td>
      <td>6500</td>
    </tr>
    <tr>
      <td>21586</td>
      <td>844000965</td>
      <td>6/26/2014</td>
      <td>224000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1500</td>
      <td>11968</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1500</td>
      <td>0.0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98010</td>
      <td>47.3095</td>
      <td>-122.002</td>
      <td>1320</td>
      <td>11303</td>
    </tr>
    <tr>
      <td>21587</td>
      <td>7852140040</td>
      <td>8/25/2014</td>
      <td>507250.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2270</td>
      <td>5536</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2270</td>
      <td>0.0</td>
      <td>2003</td>
      <td>0.0</td>
      <td>98065</td>
      <td>47.5389</td>
      <td>-121.881</td>
      <td>2270</td>
      <td>5731</td>
    </tr>
    <tr>
      <td>21595</td>
      <td>291310100</td>
      <td>1/16/2015</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>0.0</td>
      <td>2004</td>
      <td>0.0</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410</td>
      <td>1287</td>
    </tr>
  </tbody>
</table>
<p>2353 rows × 21 columns</p>
</div>




```python
# fix waterfront nulls by making nulls into 0.0 for nonwaterfront
df['waterfront'].fillna(0.0, inplace=True)
df.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront          0
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3804
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64



### Fix view nulls


```python
# number of value counts of 'view'. There's 63 null vals in 'view', time to fill them in as 0
df['view'].value_counts()
```




    0.0    19253
    2.0      956
    3.0      505
    1.0      329
    4.0      314
    Name: view, dtype: int64




```python
df['view'].fillna(0.0, inplace=True)
df.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront          0
    view                0
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3804
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64



### Fix yr_renovated nulls


```python
#looking for nulls in yr renovated, seeing that median looks strange
df['yr_renovated'].describe()
```




    count    17616.000000
    mean        83.847241
    std        400.436625
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max       2015.000000
    Name: yr_renovated, dtype: float64




```python
df.loc[df['yr_renovated'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>12</td>
      <td>114101516</td>
      <td>5/28/2014</td>
      <td>310000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1430</td>
      <td>19901</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1430</td>
      <td>0.0</td>
      <td>1927</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7558</td>
      <td>-122.229</td>
      <td>1780</td>
      <td>12697</td>
    </tr>
    <tr>
      <td>23</td>
      <td>8091400200</td>
      <td>5/16/2014</td>
      <td>252700.0</td>
      <td>2</td>
      <td>1.50</td>
      <td>1070</td>
      <td>9643</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1070</td>
      <td>0.0</td>
      <td>1985</td>
      <td>NaN</td>
      <td>98030</td>
      <td>47.3533</td>
      <td>-122.166</td>
      <td>1220</td>
      <td>8386</td>
    </tr>
    <tr>
      <td>26</td>
      <td>1794500383</td>
      <td>6/26/2014</td>
      <td>937000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>2450</td>
      <td>2691</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1750</td>
      <td>700.0</td>
      <td>1915</td>
      <td>NaN</td>
      <td>98119</td>
      <td>47.6386</td>
      <td>-122.360</td>
      <td>1760</td>
      <td>3573</td>
    </tr>
    <tr>
      <td>28</td>
      <td>5101402488</td>
      <td>6/24/2014</td>
      <td>438000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1520</td>
      <td>6380</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>790</td>
      <td>730.0</td>
      <td>1948</td>
      <td>NaN</td>
      <td>98115</td>
      <td>47.6950</td>
      <td>-122.304</td>
      <td>1520</td>
      <td>6235</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21576</td>
      <td>1931300412</td>
      <td>4/16/2015</td>
      <td>475000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1190</td>
      <td>1200</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1190</td>
      <td>0.0</td>
      <td>2008</td>
      <td>NaN</td>
      <td>98103</td>
      <td>47.6542</td>
      <td>-122.346</td>
      <td>1180</td>
      <td>1224</td>
    </tr>
    <tr>
      <td>21577</td>
      <td>8672200110</td>
      <td>3/17/2015</td>
      <td>1090000.0</td>
      <td>5</td>
      <td>3.75</td>
      <td>4170</td>
      <td>8142</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>10</td>
      <td>4170</td>
      <td>0.0</td>
      <td>2006</td>
      <td>NaN</td>
      <td>98056</td>
      <td>47.5354</td>
      <td>-122.181</td>
      <td>3030</td>
      <td>7980</td>
    </tr>
    <tr>
      <td>21579</td>
      <td>1972201967</td>
      <td>10/31/2014</td>
      <td>520000.0</td>
      <td>2</td>
      <td>2.25</td>
      <td>1530</td>
      <td>981</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1480</td>
      <td>50.0</td>
      <td>2006</td>
      <td>NaN</td>
      <td>98103</td>
      <td>47.6533</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1282</td>
    </tr>
    <tr>
      <td>21581</td>
      <td>191100405</td>
      <td>4/21/2015</td>
      <td>1580000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3410</td>
      <td>10125</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>3410</td>
      <td>?</td>
      <td>2007</td>
      <td>NaN</td>
      <td>98040</td>
      <td>47.5653</td>
      <td>-122.223</td>
      <td>2290</td>
      <td>10125</td>
    </tr>
    <tr>
      <td>21583</td>
      <td>7202300110</td>
      <td>9/15/2014</td>
      <td>810000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>3990</td>
      <td>7838</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>3990</td>
      <td>0.0</td>
      <td>2003</td>
      <td>NaN</td>
      <td>98053</td>
      <td>47.6857</td>
      <td>-122.046</td>
      <td>3370</td>
      <td>6814</td>
    </tr>
  </tbody>
</table>
<p>3804 rows × 21 columns</p>
</div>




```python
# there are loads of 0 vals in yr renovated in addition to the nulls
df['yr_renovated'].value_counts()
```




    0.0       16876
    2014.0       73
    2003.0       31
    2013.0       31
    2007.0       30
              ...  
    1934.0        1
    1971.0        1
    1954.0        1
    1950.0        1
    1944.0        1
    Name: yr_renovated, Length: 70, dtype: int64




```python
# fill na vals with 0 because there's a large amount already showing 0, just to get rid of na vals
df['yr_renovated'].fillna(0.0, inplace=True)
df.isna().sum()
```




    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64



## Check for datatypes to convert


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21420 non-null int64
    date             21420 non-null object
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    waterfront       21420 non-null float64
    view             21420 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    sqft_basement    21420 non-null object
    yr_built         21420 non-null int64
    yr_renovated     21420 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.6+ MB



```python
# there's a boatload of ?s as objects in this column
df['sqft_basement'].value_counts()
```




    0.0       12717
    ?           452
    600.0       216
    500.0       206
    700.0       205
              ...  
    20.0          1
    935.0         1
    588.0         1
    1284.0        1
    3000.0        1
    Name: sqft_basement, Length: 304, dtype: int64




```python
df.loc[df['sqft_basement'] == '?']

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6</td>
      <td>1321400060</td>
      <td>6/27/2014</td>
      <td>257500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1715</td>
      <td>?</td>
      <td>1995</td>
      <td>0.0</td>
      <td>98003</td>
      <td>47.3097</td>
      <td>-122.327</td>
      <td>2238</td>
      <td>6819</td>
    </tr>
    <tr>
      <td>18</td>
      <td>16000397</td>
      <td>12/5/2014</td>
      <td>189000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1200</td>
      <td>9850</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1200</td>
      <td>?</td>
      <td>1921</td>
      <td>0.0</td>
      <td>98002</td>
      <td>47.3089</td>
      <td>-122.210</td>
      <td>1060</td>
      <td>5095</td>
    </tr>
    <tr>
      <td>42</td>
      <td>7203220400</td>
      <td>7/7/2014</td>
      <td>861990.0</td>
      <td>5</td>
      <td>2.75</td>
      <td>3595</td>
      <td>5639</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>3595</td>
      <td>?</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98053</td>
      <td>47.6848</td>
      <td>-122.016</td>
      <td>3625</td>
      <td>5639</td>
    </tr>
    <tr>
      <td>79</td>
      <td>1531000030</td>
      <td>3/23/2015</td>
      <td>720000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3450</td>
      <td>39683</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>3450</td>
      <td>?</td>
      <td>2002</td>
      <td>0.0</td>
      <td>98010</td>
      <td>47.3420</td>
      <td>-122.025</td>
      <td>3350</td>
      <td>39750</td>
    </tr>
    <tr>
      <td>112</td>
      <td>2525310310</td>
      <td>9/16/2014</td>
      <td>272500.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1540</td>
      <td>12600</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1160</td>
      <td>?</td>
      <td>1980</td>
      <td>0.0</td>
      <td>98038</td>
      <td>47.3624</td>
      <td>-122.031</td>
      <td>1540</td>
      <td>11656</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21442</td>
      <td>3226049565</td>
      <td>7/11/2014</td>
      <td>504600.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>2360</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1390</td>
      <td>?</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6931</td>
      <td>-122.330</td>
      <td>2180</td>
      <td>5009</td>
    </tr>
    <tr>
      <td>21447</td>
      <td>1760650900</td>
      <td>7/21/2014</td>
      <td>337500.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2330</td>
      <td>4907</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2330</td>
      <td>?</td>
      <td>2013</td>
      <td>0.0</td>
      <td>98042</td>
      <td>47.3590</td>
      <td>-122.081</td>
      <td>2300</td>
      <td>3836</td>
    </tr>
    <tr>
      <td>21473</td>
      <td>6021503707</td>
      <td>1/20/2015</td>
      <td>352500.0</td>
      <td>2</td>
      <td>2.50</td>
      <td>980</td>
      <td>1010</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>980</td>
      <td>?</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.6844</td>
      <td>-122.387</td>
      <td>980</td>
      <td>1023</td>
    </tr>
    <tr>
      <td>21519</td>
      <td>2909310100</td>
      <td>10/15/2014</td>
      <td>332000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2380</td>
      <td>5737</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2380</td>
      <td>?</td>
      <td>2010</td>
      <td>0.0</td>
      <td>98023</td>
      <td>47.2815</td>
      <td>-122.356</td>
      <td>2380</td>
      <td>5396</td>
    </tr>
    <tr>
      <td>21581</td>
      <td>191100405</td>
      <td>4/21/2015</td>
      <td>1580000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3410</td>
      <td>10125</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>3410</td>
      <td>?</td>
      <td>2007</td>
      <td>0.0</td>
      <td>98040</td>
      <td>47.5653</td>
      <td>-122.223</td>
      <td>2290</td>
      <td>10125</td>
    </tr>
  </tbody>
</table>
<p>452 rows × 21 columns</p>
</div>




```python
#in order to simplify the prob of ?s in sqft basement, along with it being in object dtype, drop column completely. 
df.drop('sqft_basement', axis=1, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
# then create entire new column, based on sqft basement being the difference of sqft living and sqft above
df['sqft_basement'] = df['sqft_living'] - df['sqft_above']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>sqft_basement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>400</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>910</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sqft basement is a new column containing ints
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21420 non-null int64
    date             21420 non-null object
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    waterfront       21420 non-null float64
    view             21420 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    yr_built         21420 non-null int64
    yr_renovated     21420 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    sqft_basement    21420 non-null int64
    dtypes: float64(8), int64(12), object(1)
    memory usage: 3.6+ MB



```python
# move sqft basement column back to original position
df = df[['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



# SCRUB


```python
# Recast data types, identify outliers, check for multicollinearity, normalize data**
# Check and cast data types

#  Check for #'s that are store as objects (df.info(),df.describe())
# when converting to #'s, look for odd values (like many 0's), or strings that can't be converted.
# Decide how to deal weird/null values (df.unique(), df.isna().sum())
# df.fillna(subset=['col_with_nulls'],'fill_value'), df.replace()
#  Check for categorical variables stored as integers.
# May be easier to tell when you make a scatter plotm or pd.plotting.scatter_matrix()
# Check for missing values (df.isna().sum())

# Can drop rows or colums
# For missing numeric data with median or bin/convert to categorical
# For missing categorical data: make NaN own category OR replace with most common category
#  Check for multicollinearity
# Use seaborn to make correlation matrix plot
# Good rule of thumb is anything over 0.75 corr is high, remove the variable that has the most correl with the largest # of variables
#  Normalize data (may want to do after some exploring)
# Most popular is Z-scoring (but won't fix skew)
# Can log-transform to fix skewed data
```


```python
# reviewing describe to see 33 max value for bedrooms looks weird
# year renovated has min val as zero, so we'll take a look at that
# it looks like there's a few categorical variables we can touch on
display(df.describe().round(3))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.142000e+04</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>4.580940e+09</td>
      <td>540739.304</td>
      <td>3.374</td>
      <td>2.118</td>
      <td>2083.133</td>
      <td>15128.038</td>
      <td>1.496</td>
      <td>0.007</td>
      <td>0.234</td>
      <td>3.411</td>
      <td>7.663</td>
      <td>1791.170</td>
      <td>291.962</td>
      <td>1971.093</td>
      <td>68.957</td>
      <td>98077.874</td>
      <td>47.560</td>
      <td>-122.214</td>
      <td>1988.384</td>
      <td>12775.718</td>
    </tr>
    <tr>
      <td>std</td>
      <td>2.876761e+09</td>
      <td>367931.110</td>
      <td>0.925</td>
      <td>0.769</td>
      <td>918.808</td>
      <td>41530.797</td>
      <td>0.540</td>
      <td>0.082</td>
      <td>0.765</td>
      <td>0.650</td>
      <td>1.172</td>
      <td>828.693</td>
      <td>442.876</td>
      <td>29.387</td>
      <td>364.552</td>
      <td>53.477</td>
      <td>0.139</td>
      <td>0.141</td>
      <td>685.537</td>
      <td>27345.622</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000102e+06</td>
      <td>78000.000</td>
      <td>1.000</td>
      <td>0.500</td>
      <td>370.000</td>
      <td>520.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>370.000</td>
      <td>0.000</td>
      <td>1900.000</td>
      <td>0.000</td>
      <td>98001.000</td>
      <td>47.156</td>
      <td>-122.519</td>
      <td>399.000</td>
      <td>651.000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2.123537e+09</td>
      <td>322500.000</td>
      <td>3.000</td>
      <td>1.750</td>
      <td>1430.000</td>
      <td>5040.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>7.000</td>
      <td>1200.000</td>
      <td>0.000</td>
      <td>1952.000</td>
      <td>0.000</td>
      <td>98033.000</td>
      <td>47.471</td>
      <td>-122.328</td>
      <td>1490.000</td>
      <td>5100.000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>3.904921e+09</td>
      <td>450000.000</td>
      <td>3.000</td>
      <td>2.250</td>
      <td>1920.000</td>
      <td>7614.000</td>
      <td>1.500</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>7.000</td>
      <td>1560.000</td>
      <td>0.000</td>
      <td>1975.000</td>
      <td>0.000</td>
      <td>98065.000</td>
      <td>47.572</td>
      <td>-122.230</td>
      <td>1840.000</td>
      <td>7620.000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>7.308900e+09</td>
      <td>645000.000</td>
      <td>4.000</td>
      <td>2.500</td>
      <td>2550.000</td>
      <td>10690.500</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4.000</td>
      <td>8.000</td>
      <td>2220.000</td>
      <td>560.000</td>
      <td>1997.000</td>
      <td>0.000</td>
      <td>98117.000</td>
      <td>47.678</td>
      <td>-122.125</td>
      <td>2370.000</td>
      <td>10086.250</td>
    </tr>
    <tr>
      <td>max</td>
      <td>9.900000e+09</td>
      <td>7700000.000</td>
      <td>33.000</td>
      <td>8.000</td>
      <td>13540.000</td>
      <td>1651359.000</td>
      <td>3.500</td>
      <td>1.000</td>
      <td>4.000</td>
      <td>5.000</td>
      <td>13.000</td>
      <td>9410.000</td>
      <td>4820.000</td>
      <td>2015.000</td>
      <td>2015.000</td>
      <td>98199.000</td>
      <td>47.778</td>
      <td>-121.315</td>
      <td>6210.000</td>
      <td>871200.000</td>
    </tr>
  </tbody>
</table>
</div>


## Drop unnecessary cols to simplify.


```python
#dropping pointless columns for ease of analysis
# drop id, date, view, soft above and basement, 
df_basic = df.drop(['id','date','view','sqft_above','sqft_basement'], axis=1)
df_basic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(df.shape)
display(df_basic.shape)
```


    (21420, 21)



    (21420, 16)


## Check for categorical variables stored as integers.


```python
df_basic.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated',
           'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'],
          dtype='object')




```python
#checking plots for categorical columns
import matplotlib.pyplot as plt
%matplotlib inline

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(df_basic)[1:5], axes):
    df_basic.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')
    
plt.tight_layout()
```


![png](output_45_0.png)



```python
# something is weird with sqft lot, looks like there are a lot of outliers.
```


```python
#checking plots for categorical columns
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(df_basic)[5:9], axes):
    df_basic.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')
    
plt.tight_layout()
```


![png](output_47_0.png)



```python
#checking plots for categorical columns
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(df_basic)[9:13], axes):
    df_basic.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')
    
plt.tight_layout()
```


![png](output_48_0.png)



```python
# yr renovated seems categorical but that's bc of the zero vals throwing off the graph incrementally. 
# zip code is categorical
```


```python
#checking plots for categorical columns

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(df_basic)[13:15], axes):
    df_basic.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')
    
plt.tight_layout()
```


![png](output_50_0.png)



```python
# sqft lot15 seems to have a lot of outliers. 
```


```python
# categories - don’t log these - (use one hot c shortcut)
# 	waterfront - already in 1 and 0’s
# 	condition
# 	zipcode
# 	yr_built 
# 	yr renovated
	

# numerical
# 	bedrooms (need to filter)
# 	bathrooms (need to filter)
# 	floors
# 	grade (ordinal category so can keep numeric - need to scale though)
# 	soft living (need to filter)
# 	soft lot (need to filter)
# 	lat (maybe to filter, will check after first model)
# 	long (maybe to filter, will check after first model)
# 	sqft lot 15 (need to filter)
```


```python
# list to look over:
# need to change list of categories, will try model before and after to compare.
# sqft lot, looks like there are a lot of outliers
# yr renovated seems categorical but that's bc of the zero vals throwing off the graph incrementally. 
# sqft lot15 seems to have a lot of outliers. 
```

### When converting to #'s, look for odd values (like many 0's), or strings that can't be converted.


```python
# # year renovated has min val as zero, so we'll take a look at that
# df.plot(kind='scatter', x='yr_renovated', y='price', alpha=0.4, color='b');
```


```python
# df.head(20)
```


```python
# sorted(df['yr_renovated'].unique())
```


```python
# df.loc[df['yr_renovated'] > 0]. describe()
```


```python
# # we are going to put yr renovated into bins and then into categories
# df['yr_renovated'].describe() 
```

### Bin/convert yr_renovated to categorical.


```python
# # First, create bins based on the values observed. 5 values will result in 4 bins
# bins = [-1, 1930, 1950 , 1970, 1990, 2010, 2020]

# # Use pd.cut()
# bins_yr_renovated = pd.cut(df['yr_renovated'], bins)
```


```python
# # Using pd.cut() returns unordered categories. Transform this to ordered categories 
# bins_yr_renovated = bins_yr_renovated.cat.as_ordered()
# bins_yr_renovated.head()
```


```python
# # Inspect the result
# bins_yr_renovated.value_counts().plot(kind='bar')
```


```python
# # Replace the existing 'yr_renovated' column
# df['yr_renovated']=bins_yr_renovated
```


```python
# df.head()
```


```python
# # label encode yr renovated for regression
# # [(-1, 1930] < (1930, 1950] < (1950, 1970] < (1970, 1990] < (1990, 2010] < (2010, 2020]]
# # 0 = never renovated, 1=1930-50, 2=1950-70, 3=1970-90, 4=1990-2010, 5=2010-present

# df['yr_renovated'] = df['yr_renovated'].cat.codes
```


```python
# # 0 = never renovated, 1=1930-50, 2=1950-70, 3=1970-90, 4=1990-2010, 5=2010-present
# df['yr_renovated'].value_counts()
```


```python
# # notice the difference now compared to the previous renovated scatter. much more organized.
# df.plot(kind='scatter', x='yr_renovated', y='price', alpha=0.4, color='b');
```

## Check for multicollinearity.

### Create heatmap for collinearity check. 


```python
fig, axes = plt.subplots(figsize=(13,13))
sns.heatmap(df_basic.corr().round(3), center=0, square=True, ax=axes, annot=True);
axes.set_ylim(len(df_basic.corr()),-0.5,+0.5)
```




    (16, -0.5)




![png](output_71_1.png)


Sqft_living15 seems to have a bit of correlation with sqft_living and with grade and since those two predictors already are fairly correlated, I'll drop sqft_living15. 

## Drop cols affecting multicollinearity.


```python
df_basic.drop('sqft_living15', axis=1, inplace=True)
```


```python
df_basic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



# EXPLORE


```python
# EXPLORE

#  Check distributions, outliers, etc**
#  Check scales, ranges (df.describe())
#  Check histograms to get an idea of distributions (df.hist()) and data transformations to perform.
# Can also do kernel density estimates
#  Use scatter plots to check for linearity and possible categorical variables (df.plot("x","y"))
# categoricals will look like vertical lines
#  Use pd.plotting.scatter_matrix(df) to visualize possible relationships
#  Check for linearity.
```

## Check distributions for normality.


```python
#  Normalize data (may want to do after some exploring)
df_basic.hist(figsize=(10,10));
```


![png](output_79_0.png)



```python
# normalization thoughts:
# the only continuous variables I have are sqft living, sqft lot, living 15, lot 15. 
# The rest are basically categorical. Not sure if those need to be normalized. 
```

## Data transformations 1 - Z-scoring.


```python
def find_outliers(col):
    """Use scipy to calcualte absoliute Z-scores 
    and return boolean series where True indicates it is an outlier
    Args:
        col (Series): a series/column from your DataFrame
    Returns:
        idx_outliers (Series): series of  True/False for each row in col
        
    Ex:
    >> idx_outs = find_outliers(df['bedrooms'])
    >> df_clean = df.loc[idx_outs==False]"""
    from scipy import stats
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>3,True,False)
    return pd.Series(idx_outliers,index=col.index)
```

### Normalize data by filtering outliers with z-scores.


```python
df_outliers = pd.DataFrame()

cols_to_clean = ['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_lot15']

for col in cols_to_clean:
    df_outliers[col] = find_outliers(df_basic[col])

df_outliers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_outs = df_outliers.any(axis=1)
```


```python
np.sum(test_outs)
```




    838




```python
df_cleaned = df_basic.loc[test_outs==False]
df_cleaned.describe().round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
      <td>20582.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>519332.688</td>
      <td>3.342</td>
      <td>2.076</td>
      <td>2017.215</td>
      <td>10290.621</td>
      <td>1.488</td>
      <td>0.006</td>
      <td>3.415</td>
      <td>7.609</td>
      <td>1970.639</td>
      <td>67.885</td>
      <td>98078.944</td>
      <td>47.561</td>
      <td>-122.218</td>
      <td>9456.817</td>
    </tr>
    <tr>
      <td>std</td>
      <td>307890.682</td>
      <td>0.861</td>
      <td>0.711</td>
      <td>803.745</td>
      <td>12044.713</td>
      <td>0.540</td>
      <td>0.076</td>
      <td>0.651</td>
      <td>1.106</td>
      <td>29.431</td>
      <td>361.807</td>
      <td>53.688</td>
      <td>0.138</td>
      <td>0.138</td>
      <td>9421.736</td>
    </tr>
    <tr>
      <td>min</td>
      <td>78000.000</td>
      <td>1.000</td>
      <td>0.500</td>
      <td>370.000</td>
      <td>520.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>1900.000</td>
      <td>0.000</td>
      <td>98001.000</td>
      <td>47.156</td>
      <td>-122.512</td>
      <td>651.000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>319950.000</td>
      <td>3.000</td>
      <td>1.500</td>
      <td>1410.000</td>
      <td>5000.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>7.000</td>
      <td>1951.000</td>
      <td>0.000</td>
      <td>98033.000</td>
      <td>47.474</td>
      <td>-122.331</td>
      <td>5035.250</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>445000.000</td>
      <td>3.000</td>
      <td>2.250</td>
      <td>1890.000</td>
      <td>7497.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>7.000</td>
      <td>1974.000</td>
      <td>0.000</td>
      <td>98070.000</td>
      <td>47.573</td>
      <td>-122.237</td>
      <td>7500.000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>630000.000</td>
      <td>4.000</td>
      <td>2.500</td>
      <td>2500.000</td>
      <td>10200.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>4.000</td>
      <td>8.000</td>
      <td>1996.000</td>
      <td>0.000</td>
      <td>98118.000</td>
      <td>47.679</td>
      <td>-122.132</td>
      <td>9799.250</td>
    </tr>
    <tr>
      <td>max</td>
      <td>3640000.000</td>
      <td>6.000</td>
      <td>4.250</td>
      <td>4830.000</td>
      <td>137214.000</td>
      <td>3.500</td>
      <td>1.000</td>
      <td>5.000</td>
      <td>13.000</td>
      <td>2015.000</td>
      <td>2015.000</td>
      <td>98199.000</td>
      <td>47.778</td>
      <td>-121.315</td>
      <td>93825.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



### Showing bedrooms before and after filter.


```python
# 'bedrooms','bathrooms','sqft_living','sqft_lot','yr_built','sqft_lot15'
```


```python
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
df_basic.plot(x='bedrooms', y='price', ax=ax1, kind='scatter')
ax1.set_title("bedrooms before filter")

ax2 = plt.subplot(1, 2, 2)
df_cleaned.plot(x='bedrooms', y='price', ax=ax2, kind='scatter')
ax2.set_title('bedrooms after filter')
plt.subplots_adjust(wspace=0.4)
```


![png](output_91_0.png)



```python
# so it looks like my bedrooms are filtered from 1 to 6 bedrooms now
```

### Showing bathrooms before and after filter.


```python
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
df_basic.plot(x='bathrooms', y='price', ax=ax1, kind='scatter')
ax1.set_title("bathrooms before filter")

ax2 = plt.subplot(1, 2, 2)
df_cleaned.plot(x='bathrooms', y='price', ax=ax2, kind='scatter')
ax2.set_title('bathrooms after filter')
plt.subplots_adjust(wspace=0.4)
```


![png](output_94_0.png)



```python
# it looks like the z score filter put my bathroom range from .5 to 4.5. sweet. 
```

### Showing sqft living area before and after filter.


```python
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
df_basic.plot(x='sqft_living', y='price', ax=ax1, kind='scatter')
ax1.set_title("square foot living area before filter")

ax2 = plt.subplot(1, 2, 2)
df_cleaned.plot(x='sqft_living', y='price', ax=ax2, kind='scatter')
ax2.set_title('square foot living area after filter')
plt.subplots_adjust(wspace=0.4)
```


![png](output_97_0.png)



```python
#it looks like the zscore filtered my sqft living from about 500 to 4800
```

### Showing sqft lot before and after filter.


```python
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
df_basic.plot(x='sqft_lot', y='price', ax=ax1, kind='scatter')
ax1.set_title("square foot lot area before filter")

ax2 = plt.subplot(1, 2, 2)
df_cleaned.plot(x='sqft_lot', y='price', ax=ax2, kind='scatter')
ax2.set_title('square foot lot area after filter')
plt.subplots_adjust(wspace=0.4)
```


![png](output_100_0.png)



```python
# it looks like the zscore filtered my sqft lot from 0 to 140000, way better than the outlier 1.6 million
```

### Showing sqft lot 15 before and after filter.


```python
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
df_basic.plot(x='sqft_lot15', y='price', ax=ax1, kind='scatter')
ax1.set_title("square foot lot area 15 before filter")

ax2 = plt.subplot(1, 2, 2)
df_cleaned.plot(x='sqft_lot15', y='price', ax=ax2, kind='scatter')
ax2.set_title('square foot lot area 15 after filter')
plt.subplots_adjust(wspace=0.4)
```


![png](output_103_0.png)


### Check model normality after zscore filter outliers. 


```python
df_cleaned.hist(figsize=(9,9));
```


![png](output_105_0.png)


## Create model.

### Create model with data as is, not accounting for categorical data. 

First we create a model with variables as is, filtered but not accounting for some data needing to be categorized. Following up with a model below that will show the difference in output based on creating categorical variables where necessary. 


```python
# create model using list of predictors by typing them all out.
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated',
       'zipcode', 'lat', 'long', 'sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df_cleaned).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.676</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.676</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   3062.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 06 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:56:55</td>     <th>  Log-Likelihood:    </th> <td>-2.7772e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20582</td>      <th>  AIC:               </th>  <td>5.555e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20567</td>      <th>  BIC:               </th>  <td>5.556e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    14</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td> 7.406e+06</td> <td> 2.53e+06</td> <td>    2.931</td> <td> 0.003</td> <td> 2.45e+06</td> <td> 1.24e+07</td>
</tr>
<tr>
  <th>bedrooms</th>     <td>-3.063e+04</td> <td> 1854.500</td> <td>  -16.514</td> <td> 0.000</td> <td>-3.43e+04</td> <td> -2.7e+04</td>
</tr>
<tr>
  <th>bathrooms</th>    <td> 3.359e+04</td> <td> 2997.685</td> <td>   11.205</td> <td> 0.000</td> <td> 2.77e+04</td> <td> 3.95e+04</td>
</tr>
<tr>
  <th>sqft_living</th>  <td>  153.6617</td> <td>    3.041</td> <td>   50.524</td> <td> 0.000</td> <td>  147.700</td> <td>  159.623</td>
</tr>
<tr>
  <th>sqft_lot</th>     <td>    0.1952</td> <td>    0.177</td> <td>    1.101</td> <td> 0.271</td> <td>   -0.152</td> <td>    0.543</td>
</tr>
<tr>
  <th>floors</th>       <td>  1.66e+04</td> <td> 2910.874</td> <td>    5.703</td> <td> 0.000</td> <td> 1.09e+04</td> <td> 2.23e+04</td>
</tr>
<tr>
  <th>waterfront</th>   <td> 6.804e+05</td> <td> 1.64e+04</td> <td>   41.612</td> <td> 0.000</td> <td> 6.48e+05</td> <td> 7.12e+05</td>
</tr>
<tr>
  <th>condition</th>    <td> 2.922e+04</td> <td> 2088.975</td> <td>   13.986</td> <td> 0.000</td> <td> 2.51e+04</td> <td> 3.33e+04</td>
</tr>
<tr>
  <th>grade</th>        <td> 1.139e+05</td> <td> 1843.140</td> <td>   61.807</td> <td> 0.000</td> <td>  1.1e+05</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>yr_built</th>     <td>-2680.7818</td> <td>   64.270</td> <td>  -41.711</td> <td> 0.000</td> <td>-2806.756</td> <td>-2554.808</td>
</tr>
<tr>
  <th>yr_renovated</th> <td>   28.4692</td> <td>    3.566</td> <td>    7.983</td> <td> 0.000</td> <td>   21.479</td> <td>   35.459</td>
</tr>
<tr>
  <th>zipcode</th>      <td> -469.1360</td> <td>   29.099</td> <td>  -16.122</td> <td> 0.000</td> <td> -526.172</td> <td> -412.100</td>
</tr>
<tr>
  <th>lat</th>          <td> 5.757e+05</td> <td> 9520.914</td> <td>   60.468</td> <td> 0.000</td> <td> 5.57e+05</td> <td> 5.94e+05</td>
</tr>
<tr>
  <th>long</th>         <td> -1.29e+05</td> <td> 1.17e+04</td> <td>  -11.010</td> <td> 0.000</td> <td>-1.52e+05</td> <td>-1.06e+05</td>
</tr>
<tr>
  <th>sqft_lot15</th>   <td>   -1.2844</td> <td>    0.230</td> <td>   -5.580</td> <td> 0.000</td> <td>   -1.736</td> <td>   -0.833</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11149.879</td> <th>  Durbin-Watson:     </th>  <td>   1.992</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>185029.575</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.238</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>16.990</td>   <th>  Cond. No.          </th>  <td>2.05e+08</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.05e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# You'll notice that our R2 value is 0.676 which is eh.

# The P value for sqft_lot is at 0.271, well above the accepted 0.05 so that may have some issues with collinearity. 
# We can check the heat map on that one again. 

# Our skew is 2.2.
# Kurtosis is 16.99.
# Condition is 205,000,000
```

### Create model using numerical and categorical predictors.

In order to be sure our categorical variables are not being used in regression as values instead of labels, we will use stats to use a C() operator which will essentially one hot those specific variables


```python
# create model using list of predictors by typing them all out. 
# removed sqft lot
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'floors',
       'C(waterfront)', 'condition', 'grade', 'yr_built', 'yr_renovated',
       'C(zipcode)', 'lat', 'long', 'sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = smf.ols(formula=formula, data=df_cleaned).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.798</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.797</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   998.5</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 06 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:56:56</td>     <th>  Log-Likelihood:    </th> <td>-2.7286e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20582</td>      <th>  AIC:               </th>  <td>5.459e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20500</td>      <th>  BIC:               </th>  <td>5.465e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    81</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>            <td> -1.94e+07</td> <td> 5.68e+06</td> <td>   -3.414</td> <td> 0.001</td> <td>-3.05e+07</td> <td>-8.26e+06</td>
</tr>
<tr>
  <th>C(waterfront)[T.1.0]</th> <td>  7.24e+05</td> <td> 1.32e+04</td> <td>   54.863</td> <td> 0.000</td> <td> 6.98e+05</td> <td>  7.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>  <td> 3.082e+04</td> <td> 1.27e+04</td> <td>    2.433</td> <td> 0.015</td> <td> 5989.164</td> <td> 5.57e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>  <td>-1.959e+04</td> <td> 1.13e+04</td> <td>   -1.734</td> <td> 0.083</td> <td>-4.17e+04</td> <td> 2560.321</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>  <td> 6.869e+05</td> <td> 2.13e+04</td> <td>   32.296</td> <td> 0.000</td> <td> 6.45e+05</td> <td> 7.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>  <td> 2.829e+05</td> <td> 2.26e+04</td> <td>   12.539</td> <td> 0.000</td> <td> 2.39e+05</td> <td> 3.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>  <td> 2.556e+05</td> <td> 1.86e+04</td> <td>   13.774</td> <td> 0.000</td> <td> 2.19e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>  <td> 2.142e+05</td> <td> 2.33e+04</td> <td>    9.173</td> <td> 0.000</td> <td> 1.68e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>  <td>  2.34e+05</td> <td> 2.22e+04</td> <td>   10.542</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.78e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>  <td> 1.126e+05</td> <td> 2.04e+04</td> <td>    5.526</td> <td> 0.000</td> <td> 7.26e+04</td> <td> 1.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>  <td> 8.329e+04</td> <td>  2.9e+04</td> <td>    2.874</td> <td> 0.004</td> <td> 2.65e+04</td> <td>  1.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>  <td> 1.284e+05</td> <td> 3.35e+04</td> <td>    3.834</td> <td> 0.000</td> <td> 6.28e+04</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>  <td> 8.651e+04</td> <td> 3.16e+04</td> <td>    2.733</td> <td> 0.006</td> <td> 2.45e+04</td> <td> 1.49e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>  <td> 5.917e+04</td> <td> 1.79e+04</td> <td>    3.305</td> <td> 0.001</td> <td> 2.41e+04</td> <td> 9.43e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>  <td>-5.078e+04</td> <td> 1.04e+04</td> <td>   -4.862</td> <td> 0.000</td> <td>-7.12e+04</td> <td>-3.03e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>  <td> 1.651e+05</td> <td> 3.07e+04</td> <td>    5.383</td> <td> 0.000</td> <td> 1.05e+05</td> <td> 2.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>  <td> 1.699e+05</td> <td> 1.91e+04</td> <td>    8.899</td> <td> 0.000</td> <td> 1.32e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>  <td> 7.449e+04</td> <td> 2.82e+04</td> <td>    2.643</td> <td> 0.008</td> <td> 1.92e+04</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>  <td> 2.107e+05</td> <td> 2.18e+04</td> <td>    9.675</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>  <td> 6512.4171</td> <td> 1.25e+04</td> <td>    0.519</td> <td> 0.603</td> <td>-1.81e+04</td> <td> 3.11e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>  <td>  1.07e+04</td> <td> 1.31e+04</td> <td>    0.818</td> <td> 0.413</td> <td>-1.49e+04</td> <td> 3.63e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>  <td>-1.583e+04</td> <td>  1.5e+04</td> <td>   -1.052</td> <td> 0.293</td> <td>-4.53e+04</td> <td> 1.37e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>  <td> 3.318e+05</td> <td> 2.42e+04</td> <td>   13.726</td> <td> 0.000</td> <td> 2.84e+05</td> <td> 3.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>  <td>  1.64e+05</td> <td>  2.6e+04</td> <td>    6.319</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 2.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>  <td> 6.028e+04</td> <td> 1.43e+04</td> <td>    4.207</td> <td> 0.000</td> <td> 3.22e+04</td> <td> 8.84e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>  <td> 1.039e+06</td> <td> 3.07e+04</td> <td>   33.886</td> <td> 0.000</td> <td> 9.79e+05</td> <td>  1.1e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>  <td>  4.74e+05</td> <td> 1.87e+04</td> <td>   25.302</td> <td> 0.000</td> <td> 4.37e+05</td> <td> 5.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>  <td> 2.016e+04</td> <td> 1.21e+04</td> <td>    1.664</td> <td> 0.096</td> <td>-3585.111</td> <td> 4.39e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>  <td> 1.578e+05</td> <td>  2.7e+04</td> <td>    5.836</td> <td> 0.000</td> <td> 1.05e+05</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>  <td> 2.066e+05</td> <td> 2.47e+04</td> <td>    8.366</td> <td> 0.000</td> <td> 1.58e+05</td> <td> 2.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>  <td> 2.087e+05</td> <td> 2.67e+04</td> <td>    7.818</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 2.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>  <td>  3.46e+04</td> <td> 1.47e+04</td> <td>    2.356</td> <td> 0.018</td> <td> 5813.545</td> <td> 6.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>  <td> 9.312e+04</td> <td>  1.6e+04</td> <td>    5.806</td> <td> 0.000</td> <td> 6.17e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>  <td>  2.23e+04</td> <td> 1.39e+04</td> <td>    1.601</td> <td> 0.109</td> <td>-4998.813</td> <td> 4.96e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>  <td>  9.31e+04</td> <td> 1.57e+04</td> <td>    5.912</td> <td> 0.000</td> <td> 6.22e+04</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>  <td>   1.5e+05</td> <td> 2.47e+04</td> <td>    6.070</td> <td> 0.000</td> <td> 1.02e+05</td> <td> 1.98e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>  <td>-3.527e+04</td> <td> 2.06e+04</td> <td>   -1.711</td> <td> 0.087</td> <td>-7.57e+04</td> <td> 5124.702</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>  <td> 1.199e+05</td> <td>  2.9e+04</td> <td>    4.139</td> <td> 0.000</td> <td> 6.31e+04</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>  <td> 1.717e+05</td> <td> 2.34e+04</td> <td>    7.344</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 2.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>  <td> 2.002e+05</td> <td> 2.24e+04</td> <td>    8.933</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 2.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>  <td> 1.037e+05</td> <td> 3.03e+04</td> <td>    3.420</td> <td> 0.001</td> <td> 4.42e+04</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>  <td>-2.187e+04</td> <td> 1.15e+04</td> <td>   -1.897</td> <td> 0.058</td> <td>-4.45e+04</td> <td>  723.783</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>  <td> 3.837e+05</td> <td> 2.47e+04</td> <td>   15.526</td> <td> 0.000</td> <td> 3.35e+05</td> <td> 4.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>  <td> 2.551e+05</td> <td> 2.33e+04</td> <td>   10.939</td> <td> 0.000</td> <td> 2.09e+05</td> <td> 3.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>  <td>  3.95e+05</td> <td> 2.39e+04</td> <td>   16.517</td> <td> 0.000</td> <td> 3.48e+05</td> <td> 4.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>  <td> 8.766e+04</td> <td> 1.71e+04</td> <td>    5.117</td> <td> 0.000</td> <td> 5.41e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>  <td> 2.553e+05</td> <td>  2.4e+04</td> <td>   10.640</td> <td> 0.000</td> <td> 2.08e+05</td> <td> 3.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>  <td> 7.371e+04</td> <td> 1.88e+04</td> <td>    3.921</td> <td> 0.000</td> <td> 3.69e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>  <td> 4.215e+05</td> <td> 2.45e+04</td> <td>   17.200</td> <td> 0.000</td> <td> 3.74e+05</td> <td>  4.7e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>  <td>  5.19e+05</td> <td> 2.19e+04</td> <td>   23.656</td> <td> 0.000</td> <td> 4.76e+05</td> <td> 5.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>  <td> 2.626e+05</td> <td> 2.37e+04</td> <td>   11.063</td> <td> 0.000</td> <td> 2.16e+05</td> <td> 3.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>  <td> 2.408e+05</td> <td> 1.92e+04</td> <td>   12.574</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.78e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>  <td> 2.358e+05</td> <td>  2.4e+04</td> <td>    9.804</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>  <td> 1.343e+05</td> <td> 1.67e+04</td> <td>    8.028</td> <td> 0.000</td> <td> 1.02e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>  <td> 3.975e+05</td> <td> 2.33e+04</td> <td>   17.079</td> <td> 0.000</td> <td> 3.52e+05</td> <td> 4.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>  <td> 2.716e+05</td> <td> 2.07e+04</td> <td>   13.097</td> <td> 0.000</td> <td> 2.31e+05</td> <td> 3.12e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>  <td> 1.432e+05</td> <td> 2.57e+04</td> <td>    5.576</td> <td> 0.000</td> <td> 9.29e+04</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>  <td> 1.466e+05</td> <td> 1.75e+04</td> <td>    8.354</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.81e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>  <td> 8.865e+04</td> <td> 2.65e+04</td> <td>    3.340</td> <td> 0.001</td> <td> 3.66e+04</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>  <td> 2.021e+05</td> <td> 1.79e+04</td> <td>   11.262</td> <td> 0.000</td> <td> 1.67e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>  <td> 2.201e+05</td> <td> 1.93e+04</td> <td>   11.415</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.58e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>  <td> 8.646e+04</td> <td>  1.6e+04</td> <td>    5.393</td> <td> 0.000</td> <td>  5.5e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>  <td>  3.49e+04</td> <td> 2.14e+04</td> <td>    1.632</td> <td> 0.103</td> <td>-7005.690</td> <td> 7.68e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>  <td> 8.542e+04</td> <td> 2.76e+04</td> <td>    3.091</td> <td> 0.002</td> <td> 3.12e+04</td> <td>  1.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>  <td> 5.278e+04</td> <td> 1.46e+04</td> <td>    3.614</td> <td> 0.000</td> <td> 2.42e+04</td> <td> 8.14e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>  <td> 3.846e+04</td> <td> 1.55e+04</td> <td>    2.488</td> <td> 0.013</td> <td> 8158.190</td> <td> 6.88e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>  <td> 1.625e+05</td> <td> 2.77e+04</td> <td>    5.869</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>  <td> 2.663e+04</td> <td> 1.59e+04</td> <td>    1.670</td> <td> 0.095</td> <td>-4626.592</td> <td> 5.79e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>  <td> 1.642e+04</td> <td> 1.62e+04</td> <td>    1.012</td> <td> 0.312</td> <td>-1.54e+04</td> <td> 4.82e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>  <td> 2536.1823</td> <td> 1.22e+04</td> <td>    0.207</td> <td> 0.836</td> <td>-2.14e+04</td> <td> 2.65e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>  <td> 3.213e+05</td> <td> 2.28e+04</td> <td>   14.115</td> <td> 0.000</td> <td> 2.77e+05</td> <td> 3.66e+05</td>
</tr>
<tr>
  <th>bedrooms</th>             <td>-2.092e+04</td> <td> 1490.173</td> <td>  -14.038</td> <td> 0.000</td> <td>-2.38e+04</td> <td> -1.8e+04</td>
</tr>
<tr>
  <th>bathrooms</th>            <td> 1.627e+04</td> <td> 2403.522</td> <td>    6.768</td> <td> 0.000</td> <td> 1.16e+04</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>sqft_living</th>          <td>  158.5773</td> <td>    2.456</td> <td>   64.556</td> <td> 0.000</td> <td>  153.763</td> <td>  163.392</td>
</tr>
<tr>
  <th>floors</th>               <td>-8253.4262</td> <td> 2465.115</td> <td>   -3.348</td> <td> 0.001</td> <td>-1.31e+04</td> <td>-3421.605</td>
</tr>
<tr>
  <th>condition</th>            <td>  2.38e+04</td> <td> 1690.950</td> <td>   14.075</td> <td> 0.000</td> <td> 2.05e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>grade</th>                <td> 7.723e+04</td> <td> 1551.250</td> <td>   49.789</td> <td> 0.000</td> <td> 7.42e+04</td> <td> 8.03e+04</td>
</tr>
<tr>
  <th>yr_built</th>             <td>-1004.8356</td> <td>   56.648</td> <td>  -17.738</td> <td> 0.000</td> <td>-1115.871</td> <td> -893.800</td>
</tr>
<tr>
  <th>yr_renovated</th>         <td>   22.0582</td> <td>    2.840</td> <td>    7.768</td> <td> 0.000</td> <td>   16.492</td> <td>   27.624</td>
</tr>
<tr>
  <th>lat</th>                  <td> 1.033e+05</td> <td>  5.8e+04</td> <td>    1.783</td> <td> 0.075</td> <td>-1.03e+04</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>long</th>                 <td>-1.299e+05</td> <td> 4.21e+04</td> <td>   -3.087</td> <td> 0.002</td> <td>-2.12e+05</td> <td>-4.74e+04</td>
</tr>
<tr>
  <th>sqft_lot15</th>           <td>    0.5823</td> <td>    0.127</td> <td>    4.570</td> <td> 0.000</td> <td>    0.333</td> <td>    0.832</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12915.892</td> <th>  Durbin-Watson:     </th>  <td>   1.983</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>410916.210</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.513</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>24.305</td>   <th>  Cond. No.          </th>  <td>7.95e+07</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.95e+07. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# In this model there is an R2 value of 0.798. Much better fit.  

# The P values in 'zipcode' 98003, 98030-98032, 98042, 98058, 98070, 98092, 
# 98178, 98188, 98198 all are well over 0.05. 
# This could be because of lack of data from housing in those zipcodes. 

# Our skew is up to 2.5, from 2.2 in the last model. Interesting. 
# Kurtosis is up to 24.3 from 16.99 in the last model.
# The condition number is down to 79,500,000 from 205,000,000 in the last model. 
```

From the results of both models, there was a clear improvement overall in ability to predict housing prices. My next thoughts would be to drop some of the columns causing multicollinearity in addition to scaling some of the data units to match up better. 

## Data transformations 2 - feature scaling.

### Normalize with min max feature scaling to manage the difference in magnitude. 

We are going to look at features that have units of measure that are very different from most of the other variables and look at scaling them to match better for a better fit model. One of the ways we can do this is by using min max scaling to adjust the magnitude of the units down to range 0-1.


```python
# Take a look at the histogram to show the difference in magnitude of our numerical features. 
df_cleaned.hist(figsize=(12,12));
```


![png](output_119_0.png)



```python
# The features we'll want to scale will be:
# bathroom, bedrooms, condition, floors, grade, lat, long, sqft living, sqft lot, sqft lot 15
```


```python
df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cleaned.shape
```




    (20582, 15)




```python
df_scaled = df_cleaned.copy()
df_scaled.shape
```




    (20582, 15)




```python
# bathroom, bedrooms, condition, floors, grade, lat, long, sqft living, sqft lot, sqft lot 15
```


```python
# this loops through the cols needing to be scaled 
# applies equation for min max scaling and adds new col to dataframe

scale = ['bedrooms','bathrooms','condition','floors','grade','sqft_living', 'sqft_lot', 'lat', 'long', 'sqft_lot15']
for col in scale:
    x = df_scaled[col]
    scaled_x = (x - min(x)) / (max(x) - min(x))
    df_scaled['scaled_' + col] = scaled_x

df_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
      <th>scaled_sqft_living</th>
      <th>scaled_sqft_lot</th>
      <th>scaled_lat</th>
      <th>scaled_long</th>
      <th>scaled_sqft_lot15</th>
      <th>scaled_bedrooms</th>
      <th>scaled_bathrooms</th>
      <th>scaled_condition</th>
      <th>scaled_floors</th>
      <th>scaled_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>5650</td>
      <td>0.181614</td>
      <td>0.037529</td>
      <td>0.571498</td>
      <td>0.213033</td>
      <td>0.053652</td>
      <td>0.4</td>
      <td>0.133333</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>7639</td>
      <td>0.493274</td>
      <td>0.049176</td>
      <td>0.908959</td>
      <td>0.161236</td>
      <td>0.074999</td>
      <td>0.4</td>
      <td>0.466667</td>
      <td>0.5</td>
      <td>0.4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>8062</td>
      <td>0.089686</td>
      <td>0.069352</td>
      <td>0.936143</td>
      <td>0.233083</td>
      <td>0.079539</td>
      <td>0.2</td>
      <td>0.133333</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>5000</td>
      <td>0.356502</td>
      <td>0.032774</td>
      <td>0.586939</td>
      <td>0.099415</td>
      <td>0.046676</td>
      <td>0.6</td>
      <td>0.666667</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>7503</td>
      <td>0.293722</td>
      <td>0.055306</td>
      <td>0.741354</td>
      <td>0.390142</td>
      <td>0.073540</td>
      <td>0.4</td>
      <td>0.400000</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



The scale columns are added to the existing dataframe and can then be applied into the model the same as the previous columns, just in place of the originals to show the difference in model performance. 

### Check hist to show new scaled data. 


```python
df_scaled.hist(figsize=(12,12));
```


![png](output_128_0.png)


# MODEL

## Create model. 

Next we create an updated model that has the scaled numerical features with similar magnitudes to rest of the data in place of previous ones. The features we scaled were 'bedrooms','bathrooms','condition','floors','grade','sqft_living', 'sqft_lot', 'lat', 'long', 'sqft_lot15'.


```python
df_scaled.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated',
           'zipcode', 'lat', 'long', 'sqft_lot15', 'scaled_sqft_living',
           'scaled_sqft_lot', 'scaled_lat', 'scaled_long', 'scaled_sqft_lot15',
           'scaled_bedrooms', 'scaled_bathrooms', 'scaled_condition',
           'scaled_floors', 'scaled_grade'],
          dtype='object')




```python
# create model using list of predictors by typing them all out. 
outcome = 'price'
x_cols = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft_living','scaled_sqft_lot', 'scaled_floors',
       'C(waterfront)', 'scaled_condition', 'scaled_grade', 'yr_built', 'yr_renovated',
       'C(zipcode)', 'scaled_lat', 'scaled_long', 'scaled_sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = smf.ols(formula=formula, data=df_scaled).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.798</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.798</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   990.4</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 06 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>18:19:16</td>     <th>  Log-Likelihood:    </th> <td>-2.7283e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20582</td>      <th>  AIC:               </th>  <td>5.458e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20499</td>      <th>  BIC:               </th>  <td>5.465e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    82</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>            <td> 1.622e+06</td> <td> 1.12e+05</td> <td>   14.436</td> <td> 0.000</td> <td>  1.4e+06</td> <td> 1.84e+06</td>
</tr>
<tr>
  <th>C(waterfront)[T.1.0]</th> <td> 7.258e+05</td> <td> 1.32e+04</td> <td>   55.088</td> <td> 0.000</td> <td>    7e+05</td> <td> 7.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>  <td> 3.343e+04</td> <td> 1.27e+04</td> <td>    2.642</td> <td> 0.008</td> <td> 8633.900</td> <td> 5.82e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>  <td>-1.746e+04</td> <td> 1.13e+04</td> <td>   -1.547</td> <td> 0.122</td> <td>-3.96e+04</td> <td> 4658.281</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>  <td>   6.9e+05</td> <td> 2.12e+04</td> <td>   32.490</td> <td> 0.000</td> <td> 6.48e+05</td> <td> 7.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>  <td> 2.831e+05</td> <td> 2.25e+04</td> <td>   12.570</td> <td> 0.000</td> <td> 2.39e+05</td> <td> 3.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>  <td> 2.572e+05</td> <td> 1.85e+04</td> <td>   13.883</td> <td> 0.000</td> <td> 2.21e+05</td> <td> 2.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>  <td> 2.158e+05</td> <td> 2.33e+04</td> <td>    9.257</td> <td> 0.000</td> <td>  1.7e+05</td> <td> 2.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>  <td> 2.356e+05</td> <td> 2.22e+04</td> <td>   10.633</td> <td> 0.000</td> <td> 1.92e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>  <td> 1.148e+05</td> <td> 2.03e+04</td> <td>    5.644</td> <td> 0.000</td> <td> 7.49e+04</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>  <td> 8.425e+04</td> <td> 2.89e+04</td> <td>    2.911</td> <td> 0.004</td> <td> 2.75e+04</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>  <td> 1.274e+05</td> <td> 3.34e+04</td> <td>    3.811</td> <td> 0.000</td> <td> 6.19e+04</td> <td> 1.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>  <td> 8.694e+04</td> <td> 3.16e+04</td> <td>    2.752</td> <td> 0.006</td> <td>  2.5e+04</td> <td> 1.49e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>  <td> 5.959e+04</td> <td> 1.79e+04</td> <td>    3.334</td> <td> 0.001</td> <td> 2.46e+04</td> <td> 9.46e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>  <td>-4.849e+04</td> <td> 1.04e+04</td> <td>   -4.649</td> <td> 0.000</td> <td>-6.89e+04</td> <td> -2.8e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>  <td>  1.63e+05</td> <td> 3.06e+04</td> <td>    5.322</td> <td> 0.000</td> <td> 1.03e+05</td> <td> 2.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>  <td> 1.714e+05</td> <td> 1.91e+04</td> <td>    8.994</td> <td> 0.000</td> <td> 1.34e+05</td> <td> 2.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>  <td> 7.428e+04</td> <td> 2.81e+04</td> <td>    2.640</td> <td> 0.008</td> <td> 1.91e+04</td> <td> 1.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>  <td> 2.122e+05</td> <td> 2.17e+04</td> <td>    9.758</td> <td> 0.000</td> <td>  1.7e+05</td> <td> 2.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>  <td> 7902.1660</td> <td> 1.25e+04</td> <td>    0.631</td> <td> 0.528</td> <td>-1.66e+04</td> <td> 3.24e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>  <td> 1.047e+04</td> <td>  1.3e+04</td> <td>    0.802</td> <td> 0.422</td> <td>-1.51e+04</td> <td>  3.6e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>  <td> -1.41e+04</td> <td>  1.5e+04</td> <td>   -0.938</td> <td> 0.348</td> <td>-4.36e+04</td> <td> 1.54e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>  <td> 3.335e+05</td> <td> 2.41e+04</td> <td>   13.818</td> <td> 0.000</td> <td> 2.86e+05</td> <td> 3.81e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>  <td>  1.65e+05</td> <td> 2.59e+04</td> <td>    6.367</td> <td> 0.000</td> <td> 1.14e+05</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>  <td> 6.219e+04</td> <td> 1.43e+04</td> <td>    4.347</td> <td> 0.000</td> <td> 3.41e+04</td> <td> 9.02e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>  <td> 1.043e+06</td> <td> 3.06e+04</td> <td>   34.079</td> <td> 0.000</td> <td> 9.83e+05</td> <td>  1.1e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>  <td> 4.765e+05</td> <td> 1.87e+04</td> <td>   25.475</td> <td> 0.000</td> <td>  4.4e+05</td> <td> 5.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>  <td> 2.074e+04</td> <td> 1.21e+04</td> <td>    1.715</td> <td> 0.086</td> <td>-2965.436</td> <td> 4.44e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>  <td> 1.582e+05</td> <td>  2.7e+04</td> <td>    5.861</td> <td> 0.000</td> <td> 1.05e+05</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>  <td> 2.075e+05</td> <td> 2.47e+04</td> <td>    8.416</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 2.56e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>  <td> 2.086e+05</td> <td> 2.66e+04</td> <td>    7.828</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 2.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>  <td>   3.5e+04</td> <td> 1.47e+04</td> <td>    2.387</td> <td> 0.017</td> <td> 6260.707</td> <td> 6.37e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>  <td> 9.437e+04</td> <td>  1.6e+04</td> <td>    5.894</td> <td> 0.000</td> <td>  6.3e+04</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>  <td> 2.352e+04</td> <td> 1.39e+04</td> <td>    1.691</td> <td> 0.091</td> <td>-3742.898</td> <td> 5.08e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>  <td> 9.438e+04</td> <td> 1.57e+04</td> <td>    6.003</td> <td> 0.000</td> <td> 6.36e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>  <td> 1.512e+05</td> <td> 2.47e+04</td> <td>    6.128</td> <td> 0.000</td> <td> 1.03e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>  <td>-3.618e+04</td> <td> 2.06e+04</td> <td>   -1.758</td> <td> 0.079</td> <td>-7.65e+04</td> <td> 4153.309</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>  <td> 1.164e+05</td> <td> 2.89e+04</td> <td>    4.024</td> <td> 0.000</td> <td> 5.97e+04</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>  <td> 1.735e+05</td> <td> 2.33e+04</td> <td>    7.434</td> <td> 0.000</td> <td> 1.28e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>  <td> 2.007e+05</td> <td> 2.24e+04</td> <td>    8.971</td> <td> 0.000</td> <td> 1.57e+05</td> <td> 2.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>  <td> 1.021e+05</td> <td> 3.03e+04</td> <td>    3.374</td> <td> 0.001</td> <td> 4.28e+04</td> <td> 1.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>  <td>-1.949e+04</td> <td> 1.15e+04</td> <td>   -1.693</td> <td> 0.090</td> <td> -4.2e+04</td> <td> 3074.716</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>  <td> 3.862e+05</td> <td> 2.47e+04</td> <td>   15.652</td> <td> 0.000</td> <td> 3.38e+05</td> <td> 4.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>  <td> 2.573e+05</td> <td> 2.33e+04</td> <td>   11.048</td> <td> 0.000</td> <td> 2.12e+05</td> <td> 3.03e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>  <td> 3.979e+05</td> <td> 2.39e+04</td> <td>   16.662</td> <td> 0.000</td> <td> 3.51e+05</td> <td> 4.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>  <td> 8.935e+04</td> <td> 1.71e+04</td> <td>    5.224</td> <td> 0.000</td> <td> 5.58e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>  <td> 2.574e+05</td> <td>  2.4e+04</td> <td>   10.746</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 3.04e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>  <td> 7.616e+04</td> <td> 1.88e+04</td> <td>    4.058</td> <td> 0.000</td> <td> 3.94e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>  <td> 4.245e+05</td> <td> 2.45e+04</td> <td>   17.349</td> <td> 0.000</td> <td> 3.77e+05</td> <td> 4.73e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>  <td> 5.225e+05</td> <td> 2.19e+04</td> <td>   23.847</td> <td> 0.000</td> <td>  4.8e+05</td> <td> 5.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>  <td> 2.649e+05</td> <td> 2.37e+04</td> <td>   11.181</td> <td> 0.000</td> <td> 2.18e+05</td> <td> 3.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>  <td> 2.435e+05</td> <td> 1.91e+04</td> <td>   12.734</td> <td> 0.000</td> <td> 2.06e+05</td> <td> 2.81e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>  <td> 2.382e+05</td> <td>  2.4e+04</td> <td>    9.920</td> <td> 0.000</td> <td> 1.91e+05</td> <td> 2.85e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>  <td>  1.37e+05</td> <td> 1.67e+04</td> <td>    8.196</td> <td> 0.000</td> <td> 1.04e+05</td> <td>  1.7e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>  <td> 4.004e+05</td> <td> 2.32e+04</td> <td>   17.231</td> <td> 0.000</td> <td> 3.55e+05</td> <td> 4.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>  <td> 2.743e+05</td> <td> 2.07e+04</td> <td>   13.248</td> <td> 0.000</td> <td> 2.34e+05</td> <td> 3.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>  <td> 1.449e+05</td> <td> 2.56e+04</td> <td>    5.652</td> <td> 0.000</td> <td> 9.47e+04</td> <td> 1.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>  <td> 1.492e+05</td> <td> 1.75e+04</td> <td>    8.513</td> <td> 0.000</td> <td> 1.15e+05</td> <td> 1.84e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>  <td> 8.992e+04</td> <td> 2.65e+04</td> <td>    3.393</td> <td> 0.001</td> <td>  3.8e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>  <td> 2.046e+05</td> <td> 1.79e+04</td> <td>   11.420</td> <td> 0.000</td> <td> 1.69e+05</td> <td>  2.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>  <td> 2.227e+05</td> <td> 1.93e+04</td> <td>   11.566</td> <td> 0.000</td> <td> 1.85e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>  <td> 8.836e+04</td> <td>  1.6e+04</td> <td>    5.520</td> <td> 0.000</td> <td>  5.7e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>  <td> 3.756e+04</td> <td> 2.13e+04</td> <td>    1.759</td> <td> 0.079</td> <td>-4283.572</td> <td> 7.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>  <td> 8.681e+04</td> <td> 2.76e+04</td> <td>    3.146</td> <td> 0.002</td> <td> 3.27e+04</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>  <td> 5.477e+04</td> <td> 1.46e+04</td> <td>    3.756</td> <td> 0.000</td> <td> 2.62e+04</td> <td> 8.33e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>  <td> 3.944e+04</td> <td> 1.54e+04</td> <td>    2.556</td> <td> 0.011</td> <td> 9193.964</td> <td> 6.97e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>  <td> 1.645e+05</td> <td> 2.76e+04</td> <td>    5.952</td> <td> 0.000</td> <td>  1.1e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>  <td> 2.933e+04</td> <td> 1.59e+04</td> <td>    1.842</td> <td> 0.065</td> <td>-1880.634</td> <td> 6.05e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>  <td>  1.89e+04</td> <td> 1.62e+04</td> <td>    1.166</td> <td> 0.243</td> <td>-1.29e+04</td> <td> 5.07e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>  <td> 4188.9410</td> <td> 1.22e+04</td> <td>    0.343</td> <td> 0.732</td> <td>-1.97e+04</td> <td> 2.81e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>  <td>  3.24e+05</td> <td> 2.27e+04</td> <td>   14.255</td> <td> 0.000</td> <td> 2.79e+05</td> <td> 3.69e+05</td>
</tr>
<tr>
  <th>scaled_bedrooms</th>      <td>-1.032e+05</td> <td> 7440.518</td> <td>  -13.872</td> <td> 0.000</td> <td>-1.18e+05</td> <td>-8.86e+04</td>
</tr>
<tr>
  <th>scaled_bathrooms</th>     <td>  6.19e+04</td> <td> 8999.099</td> <td>    6.879</td> <td> 0.000</td> <td> 4.43e+04</td> <td> 7.95e+04</td>
</tr>
<tr>
  <th>scaled_sqft_living</th>   <td> 7.006e+05</td> <td>  1.1e+04</td> <td>   63.876</td> <td> 0.000</td> <td> 6.79e+05</td> <td> 7.22e+05</td>
</tr>
<tr>
  <th>scaled_sqft_lot</th>      <td> 1.592e+05</td> <td> 1.93e+04</td> <td>    8.267</td> <td> 0.000</td> <td> 1.21e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>scaled_floors</th>        <td>-1.963e+04</td> <td> 6153.900</td> <td>   -3.189</td> <td> 0.001</td> <td>-3.17e+04</td> <td>-7562.947</td>
</tr>
<tr>
  <th>scaled_condition</th>     <td> 9.668e+04</td> <td> 6755.092</td> <td>   14.313</td> <td> 0.000</td> <td> 8.34e+04</td> <td>  1.1e+05</td>
</tr>
<tr>
  <th>scaled_grade</th>         <td> 7.721e+05</td> <td> 1.55e+04</td> <td>   49.854</td> <td> 0.000</td> <td> 7.42e+05</td> <td> 8.02e+05</td>
</tr>
<tr>
  <th>yr_built</th>             <td> -977.5412</td> <td>   56.652</td> <td>  -17.255</td> <td> 0.000</td> <td>-1088.583</td> <td> -866.499</td>
</tr>
<tr>
  <th>yr_renovated</th>         <td>   22.2541</td> <td>    2.835</td> <td>    7.850</td> <td> 0.000</td> <td>   16.697</td> <td>   27.811</td>
</tr>
<tr>
  <th>scaled_lat</th>           <td> 6.622e+04</td> <td>  3.6e+04</td> <td>    1.841</td> <td> 0.066</td> <td>-4284.602</td> <td> 1.37e+05</td>
</tr>
<tr>
  <th>scaled_long</th>          <td>-1.559e+05</td> <td> 5.03e+04</td> <td>   -3.100</td> <td> 0.002</td> <td>-2.54e+05</td> <td>-5.73e+04</td>
</tr>
<tr>
  <th>scaled_sqft_lot15</th>    <td>-5.277e+04</td> <td> 1.76e+04</td> <td>   -3.006</td> <td> 0.003</td> <td>-8.72e+04</td> <td>-1.84e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12906.466</td> <th>  Durbin-Watson:     </th>  <td>   1.985</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>413238.519</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.508</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>24.371</td>   <th>  Cond. No.          </th>  <td>3.13e+05</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.13e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# The R2 value is still at 0.798 so that's good. 

# Our skew is still 2.5. 
# Kurtosis is still 24.3.
# The condition number is down to 313,000 from 79,500,000 in the last model. 
```

### Look at heat map again with new numbers. 


```python
fig, axes = plt.subplots(figsize=(13,13))
sns.heatmap(df_scaled.corr().round(3), center=0, square=True, ax=axes, annot=True);
axes.set_ylim(len(df_scaled.corr()),-0.5,+0.5)
```




    (25, -0.5)




![png](output_136_1.png)



```python
# it seems like grade is somewhat correlated so maybe I could drop that column, 
# or I can try scaling it to see if that changes anything. 
```

### New model with only specific coeffs to compare.


```python
# create model using best coeffs list of predictors by typing them all out. 
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'scaled_sqft_living','scaled_sqft_lot',
       'C(waterfront)',
       'C(zipcode)', 'scaled_lat', 'scaled_long']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = smf.ols(formula=formula, data=df_scaled).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.771</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.770</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   906.1</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 06 Dec 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>19:38:21</td>     <th>  Log-Likelihood:    </th> <td>-2.7416e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20582</td>      <th>  AIC:               </th>  <td>5.485e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20505</td>      <th>  BIC:               </th>  <td>5.491e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    76</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>            <td> 1.387e+04</td> <td> 1.74e+04</td> <td>    0.797</td> <td> 0.426</td> <td>-2.03e+04</td> <td>  4.8e+04</td>
</tr>
<tr>
  <th>C(waterfront)[T.1.0]</th> <td> 7.605e+05</td> <td>  1.4e+04</td> <td>   54.367</td> <td> 0.000</td> <td> 7.33e+05</td> <td> 7.88e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>  <td> 3.702e+04</td> <td> 1.35e+04</td> <td>    2.749</td> <td> 0.006</td> <td> 1.06e+04</td> <td> 6.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>  <td> 4580.8726</td> <td>  1.2e+04</td> <td>    0.381</td> <td> 0.703</td> <td> -1.9e+04</td> <td> 2.82e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>  <td> 7.432e+05</td> <td> 2.26e+04</td> <td>   32.877</td> <td> 0.000</td> <td> 6.99e+05</td> <td> 7.87e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>  <td> 3.386e+05</td> <td>  2.4e+04</td> <td>   14.125</td> <td> 0.000</td> <td> 2.92e+05</td> <td> 3.86e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>  <td> 3.205e+05</td> <td> 1.97e+04</td> <td>   16.262</td> <td> 0.000</td> <td> 2.82e+05</td> <td> 3.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>  <td> 2.652e+05</td> <td> 2.48e+04</td> <td>   10.684</td> <td> 0.000</td> <td> 2.17e+05</td> <td> 3.14e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>  <td> 2.731e+05</td> <td> 2.36e+04</td> <td>   11.572</td> <td> 0.000</td> <td> 2.27e+05</td> <td> 3.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>  <td> 1.228e+05</td> <td> 2.17e+04</td> <td>    5.667</td> <td> 0.000</td> <td> 8.04e+04</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>  <td> 8.093e+04</td> <td> 3.08e+04</td> <td>    2.623</td> <td> 0.009</td> <td> 2.05e+04</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>  <td> 1.154e+05</td> <td> 3.57e+04</td> <td>    3.237</td> <td> 0.001</td> <td> 4.55e+04</td> <td> 1.85e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>  <td> 6.728e+04</td> <td> 3.37e+04</td> <td>    1.998</td> <td> 0.046</td> <td> 1262.751</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>  <td> 9.439e+04</td> <td>  1.9e+04</td> <td>    4.963</td> <td> 0.000</td> <td> 5.71e+04</td> <td> 1.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>  <td>-3.314e+04</td> <td> 1.11e+04</td> <td>   -2.983</td> <td> 0.003</td> <td>-5.49e+04</td> <td>-1.14e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>  <td> 1.649e+05</td> <td> 3.26e+04</td> <td>    5.050</td> <td> 0.000</td> <td> 1.01e+05</td> <td> 2.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>  <td> 1.956e+05</td> <td> 2.03e+04</td> <td>    9.629</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>  <td> 6.764e+04</td> <td>    3e+04</td> <td>    2.255</td> <td> 0.024</td> <td> 8852.904</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>  <td> 2.562e+05</td> <td> 2.31e+04</td> <td>   11.071</td> <td> 0.000</td> <td> 2.11e+05</td> <td> 3.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>  <td> 1.306e+04</td> <td> 1.33e+04</td> <td>    0.978</td> <td> 0.328</td> <td>-1.31e+04</td> <td> 3.92e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>  <td>  2.13e+04</td> <td> 1.39e+04</td> <td>    1.532</td> <td> 0.126</td> <td>-5957.666</td> <td> 4.86e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>  <td> 6938.8385</td> <td>  1.6e+04</td> <td>    0.433</td> <td> 0.665</td> <td>-2.45e+04</td> <td> 3.83e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>  <td> 3.594e+05</td> <td> 2.57e+04</td> <td>   13.975</td> <td> 0.000</td> <td> 3.09e+05</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>  <td> 1.688e+05</td> <td> 2.76e+04</td> <td>    6.111</td> <td> 0.000</td> <td> 1.15e+05</td> <td> 2.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>  <td> 6.285e+04</td> <td> 1.52e+04</td> <td>    4.124</td> <td> 0.000</td> <td>  3.3e+04</td> <td> 9.27e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>  <td> 1.108e+06</td> <td> 3.26e+04</td> <td>   33.996</td> <td> 0.000</td> <td> 1.04e+06</td> <td> 1.17e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>  <td> 5.501e+05</td> <td> 1.99e+04</td> <td>   27.688</td> <td> 0.000</td> <td> 5.11e+05</td> <td> 5.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>  <td> 3.278e+04</td> <td> 1.29e+04</td> <td>    2.544</td> <td> 0.011</td> <td> 7521.821</td> <td>  5.8e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>  <td> 1.835e+05</td> <td> 2.88e+04</td> <td>    6.376</td> <td> 0.000</td> <td> 1.27e+05</td> <td>  2.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>  <td> 2.364e+05</td> <td> 2.63e+04</td> <td>    8.998</td> <td> 0.000</td> <td> 1.85e+05</td> <td> 2.88e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>  <td> 2.024e+05</td> <td> 2.84e+04</td> <td>    7.124</td> <td> 0.000</td> <td> 1.47e+05</td> <td> 2.58e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>  <td> 4.094e+04</td> <td> 1.56e+04</td> <td>    2.621</td> <td> 0.009</td> <td> 1.03e+04</td> <td> 7.16e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>  <td> 1.028e+05</td> <td>  1.7e+04</td> <td>    6.030</td> <td> 0.000</td> <td> 6.94e+04</td> <td> 1.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>  <td>  4.47e+04</td> <td> 1.48e+04</td> <td>    3.016</td> <td> 0.003</td> <td> 1.56e+04</td> <td> 7.38e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>  <td> 1.023e+05</td> <td> 1.68e+04</td> <td>    6.102</td> <td> 0.000</td> <td> 6.94e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>  <td> 1.449e+05</td> <td> 2.63e+04</td> <td>    5.516</td> <td> 0.000</td> <td> 9.34e+04</td> <td> 1.96e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>  <td>-5.621e+04</td> <td> 2.17e+04</td> <td>   -2.584</td> <td> 0.010</td> <td>-9.88e+04</td> <td>-1.36e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>  <td> 1.253e+05</td> <td> 3.08e+04</td> <td>    4.065</td> <td> 0.000</td> <td> 6.49e+04</td> <td> 1.86e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>  <td> 2.194e+05</td> <td> 2.49e+04</td> <td>    8.822</td> <td> 0.000</td> <td> 1.71e+05</td> <td> 2.68e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>  <td> 2.454e+05</td> <td> 2.38e+04</td> <td>   10.295</td> <td> 0.000</td> <td> 1.99e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>  <td> 1.361e+05</td> <td> 3.22e+04</td> <td>    4.228</td> <td> 0.000</td> <td>  7.3e+04</td> <td> 1.99e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>  <td> 2987.3534</td> <td> 1.23e+04</td> <td>    0.244</td> <td> 0.808</td> <td>-2.11e+04</td> <td>  2.7e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>  <td> 4.605e+05</td> <td> 2.61e+04</td> <td>   17.625</td> <td> 0.000</td> <td> 4.09e+05</td> <td> 5.12e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>  <td> 2.918e+05</td> <td> 2.47e+04</td> <td>   11.823</td> <td> 0.000</td> <td> 2.43e+05</td> <td>  3.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>  <td> 4.598e+05</td> <td> 2.53e+04</td> <td>   18.171</td> <td> 0.000</td> <td>  4.1e+05</td> <td> 5.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>  <td> 8.355e+04</td> <td> 1.82e+04</td> <td>    4.588</td> <td> 0.000</td> <td> 4.79e+04</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>  <td> 2.885e+05</td> <td> 2.54e+04</td> <td>   11.351</td> <td> 0.000</td> <td> 2.39e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>  <td> 8.103e+04</td> <td>    2e+04</td> <td>    4.059</td> <td> 0.000</td> <td> 4.19e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>  <td> 4.911e+05</td> <td> 2.59e+04</td> <td>   18.941</td> <td> 0.000</td> <td>  4.4e+05</td> <td> 5.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>  <td> 6.012e+05</td> <td> 2.32e+04</td> <td>   25.963</td> <td> 0.000</td> <td> 5.56e+05</td> <td> 6.47e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>  <td> 2.945e+05</td> <td> 2.52e+04</td> <td>   11.693</td> <td> 0.000</td> <td> 2.45e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>  <td>  2.81e+05</td> <td> 2.03e+04</td> <td>   13.845</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>  <td> 2.647e+05</td> <td> 2.55e+04</td> <td>   10.378</td> <td> 0.000</td> <td> 2.15e+05</td> <td> 3.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>  <td> 1.493e+05</td> <td> 1.77e+04</td> <td>    8.415</td> <td> 0.000</td> <td> 1.15e+05</td> <td> 1.84e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>  <td> 4.659e+05</td> <td> 2.46e+04</td> <td>   18.931</td> <td> 0.000</td> <td> 4.18e+05</td> <td> 5.14e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>  <td> 3.356e+05</td> <td> 2.19e+04</td> <td>   15.319</td> <td> 0.000</td> <td> 2.93e+05</td> <td> 3.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>  <td> 1.484e+05</td> <td> 2.73e+04</td> <td>    5.437</td> <td> 0.000</td> <td> 9.49e+04</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>  <td> 1.664e+05</td> <td> 1.86e+04</td> <td>    8.933</td> <td> 0.000</td> <td>  1.3e+05</td> <td> 2.03e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>  <td> 9.373e+04</td> <td> 2.82e+04</td> <td>    3.323</td> <td> 0.001</td> <td> 3.84e+04</td> <td> 1.49e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>  <td>  2.32e+05</td> <td>  1.9e+04</td> <td>   12.189</td> <td> 0.000</td> <td> 1.95e+05</td> <td> 2.69e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>  <td>  2.52e+05</td> <td> 2.04e+04</td> <td>   12.335</td> <td> 0.000</td> <td> 2.12e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>  <td> 8.141e+04</td> <td>  1.7e+04</td> <td>    4.780</td> <td> 0.000</td> <td>  4.8e+04</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>  <td> 3.439e+04</td> <td> 2.27e+04</td> <td>    1.512</td> <td> 0.131</td> <td>-1.02e+04</td> <td>  7.9e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>  <td> 8.686e+04</td> <td> 2.94e+04</td> <td>    2.956</td> <td> 0.003</td> <td> 2.93e+04</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>  <td> 6.964e+04</td> <td> 1.55e+04</td> <td>    4.492</td> <td> 0.000</td> <td> 3.92e+04</td> <td>    1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>  <td> 2.133e+04</td> <td> 1.64e+04</td> <td>    1.299</td> <td> 0.194</td> <td>-1.08e+04</td> <td> 5.35e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>  <td> 1.872e+05</td> <td> 2.94e+04</td> <td>    6.361</td> <td> 0.000</td> <td>  1.3e+05</td> <td> 2.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>  <td> 2.435e+04</td> <td> 1.69e+04</td> <td>    1.437</td> <td> 0.151</td> <td>-8868.176</td> <td> 5.76e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>  <td> 1.615e+04</td> <td> 1.73e+04</td> <td>    0.935</td> <td> 0.350</td> <td>-1.77e+04</td> <td>    5e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>  <td> 9747.5611</td> <td>  1.3e+04</td> <td>    0.749</td> <td> 0.454</td> <td>-1.58e+04</td> <td> 3.53e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>  <td> 3.697e+05</td> <td> 2.42e+04</td> <td>   15.299</td> <td> 0.000</td> <td> 3.22e+05</td> <td> 4.17e+05</td>
</tr>
<tr>
  <th>bedrooms</th>             <td>-2.869e+04</td> <td> 1550.649</td> <td>  -18.499</td> <td> 0.000</td> <td>-3.17e+04</td> <td>-2.56e+04</td>
</tr>
<tr>
  <th>bathrooms</th>            <td> 1.436e+04</td> <td> 2204.755</td> <td>    6.512</td> <td> 0.000</td> <td>    1e+04</td> <td> 1.87e+04</td>
</tr>
<tr>
  <th>scaled_sqft_living</th>   <td> 9.958e+05</td> <td> 9842.679</td> <td>  101.170</td> <td> 0.000</td> <td> 9.76e+05</td> <td> 1.02e+06</td>
</tr>
<tr>
  <th>scaled_sqft_lot</th>      <td> 1.788e+05</td> <td> 1.37e+04</td> <td>   13.100</td> <td> 0.000</td> <td> 1.52e+05</td> <td> 2.06e+05</td>
</tr>
<tr>
  <th>scaled_lat</th>           <td> 9.701e+04</td> <td> 3.83e+04</td> <td>    2.532</td> <td> 0.011</td> <td> 2.19e+04</td> <td> 1.72e+05</td>
</tr>
<tr>
  <th>scaled_long</th>          <td>-2.368e+05</td> <td> 5.34e+04</td> <td>   -4.436</td> <td> 0.000</td> <td>-3.41e+05</td> <td>-1.32e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12337.671</td> <th>  Durbin-Watson:     </th>  <td>   1.989</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>361817.188</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.375</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>22.984</td>   <th>  Cond. No.          </th>  <td>    673.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
from scipy import stats
plt.style.use('ggplot')

f = 'price~bedrooms'
f2 = 'price~bathrooms'
f3 = 'price~scaled_sqft_living'
f4 = 'price~scaled_sqft_lot'
f5 = 'price~condition'
f6 = 'price~grade'
model = smf.ols(formula=f, data=df_scaled).fit()
model2 = smf.ols(formula=f2, data=df_scaled).fit()
model3 = smf.ols(formula=f3, data=df_scaled).fit()
model4 = smf.ols(formula=f4, data=df_scaled).fit()
model5 = smf.ols(formula=f5, data=df_scaled).fit()
model6 = smf.ols(formula=f6, data=df_scaled).fit()


resid1 = model.resid
resid2 = model2.resid
resid3 = model3.resid
resid4 = model4.resid
resid5 = model5.resid
resid6 = model6.resid

fig = sm.graphics.qqplot(resid1, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid2, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid3, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid4, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid5, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid6, dist=stats.norm, line='45', fit=True)
```


![png](output_140_0.png)



![png](output_140_1.png)



![png](output_140_2.png)



![png](output_140_3.png)



![png](output_140_4.png)



![png](output_140_5.png)


# iNTERPRET

### bedroom vs bathroom


```python
# bedrooms	 -28,690
# bathrooms	 14,360
# according to my previous model the bathrooms have a bigger coeff and more positive relationship with price.
# An additional bedroom is associated with a -28,690 decrease in price.
# An additional bathroom is associated with a 14,360 increase in price.
df_scaled.plot(kind='scatter', x='bedrooms', y='price', alpha=0.4, color='b')
df_scaled.plot(kind='scatter', x='bathrooms', y='price', alpha=0.4, color='b');
```


![png](output_143_0.png)



![png](output_143_1.png)


### sqft living vs sqft lot


```python
# scaled_sqft_living	9.958e+05
# scaled_sqft_lot	1.788e+05 
# according to model my sqft living area is more significant
# A 1% "unit" increase in sqft living is associated with a 9.9% "unit" increase in price.
# A 1% "unit" increase in sqft lot is associated with a 1.7% "unit" increase in price.

df_scaled.plot(kind='scatter', x='sqft_living', y='price', alpha=0.4, color='b')
df_scaled.plot(kind='scatter', x='sqft_lot', y='price', alpha=0.4, color='b');
```


![png](output_145_0.png)



![png](output_145_1.png)


### Waterfront property


```python
# waterfront coeff 760,500
# having a property on a waterfront equates to an increase in price of about 760,500
df_scaled.plot(kind='scatter', x='waterfront', y='price', alpha=0.4, color='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c256a2588>




![png](output_147_1.png)


### Latitude and Longitude


```python
# scaled_lat	9.701e+04
# scaled_long	-2.368e+05
# city center latitude 47.608013 : 1% closer to city center = 9.7% increase in price
# city center longitude -122.335167 : 1% closer to city center = 2.3% increase in price

df_scaled.plot(kind='scatter', x='lat', y='price', alpha=0.4, color='b')
df_scaled.plot(kind='scatter', x='long', y='price', alpha=0.4, color='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c30c710f0>




![png](output_149_1.png)



![png](output_149_2.png)


# CONCLUSIONS & RECOMMENDATIONS

> Best predictors for house pricing:
    - # of bathrooms (recommend adding a bathroom to your home)
    - square footage of living area (recommend finding most sqft you can afford)
    - find a property on the waterfront
    - find a property closer to city center



```python

```
