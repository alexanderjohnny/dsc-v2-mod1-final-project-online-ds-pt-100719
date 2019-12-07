
# Final Project Submission

Please fill out:
* Student name: alex beat
* Student pace: self paced / part time / full time: part time
* Scheduled project review date/time: na
* Instructor name: James Irving
* Blog post URL: na
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



## RESOURCES FOR YOU 
**<font color='red'>(Delete from final notebook)</font>**

- [OVERVIEW OF OSEMiN](#OVERVIEW-OF-OSEMiN)
- [PROCESS-CHECKLIST](#PROCESS-CHECKLIST)
    - Can actually keep this part if you'd like.
- [LINKS FOR MOD 1 PROJECT](#LINKS-FOR-MOD-1-PROJECT)



# RESOURCES FOR YOU
<font color='red' weight='bold'>- NOTE: DELETE THIS SECTION & SUB SECTIONS FROM YOUR FINAL NOTEBOOK</font>

### LINKS FOR MOD 1 PROJECT
* [Blog Post: 5 steps of a data science project lifecycle](https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492)
* ML models google sheet https://docs.google.com/spreadsheets/d/1qe4nYjGKSxBNCkeV2gxxgObBpKKc0TKrbk0Y9LTrpV8
* How to detect and remove outliers:
    * https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
* How to handle categorical variables. 
    * https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
* Sci-Kit Learn’s Scalers visually explained 
    * http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html


## OVERVIEW OF OSEMiN

<img src='https://raw.githubusercontent.com/jirvingphd/fsds_100719_cohort_notes/master/images/OSEMN.png' width=800>
<center><a href="https://www.kdnuggets.com/2018/02/data-science-command-line-book-exploring-data.html"> 
    </a></center>

> <font size=4em>The Data Science Process we'll be using during this section--OSEMiN (pronounced "OH-sum", rhymes with "possum").  This is the most straightforward of the Data Science Processes discussed so far.  **Note that during this process, just like the others, the stages often blur together.***  It is completely acceptable (and ***often a best practice!) to float back and forth** between stages as you learn new things about your problem, dataset, requirements, etc.  
It's quite common to get to the modeling step and realize that you need to scrub your data a bit more or engineer a different feature and jump back to the "Scrub" stage, or go all the way back to the "Obtain" stage when you realize your current data isn't sufficient to solve this problem. 
As with any of these frameworks, *OSEMiN is meant to be treated as guidelines, not law. 
</font>


### OSEMN DETAILS


**OBTAIN**

- This step involves understanding stakeholder requirements, gathering information on the problem, and finally sourcing data that we think will be necessary for solving this problem. 

**SCRUB**

- During this stage, we'll focus on preprocessing our data.  Important steps such as identifying and removing null values, dealing with outliers, normalizing data, and feature engineering/feature selection are handled around this stage.  The line with this stage really blurs with the _Explore_ stage, as it is common to only realize that certain columns require cleaning or preprocessing as a result of the visualzations and explorations done during Step 3.  

- Note that although technically, categorical data should be one-hot encoded during this step, in practice, it's usually done after data exploration.  This is because it is much less time-consuming to visualize and explore a few columns containing categorical data than it is to explore many different dummy columns that have been one-hot encoded. 

**EXPLORE**

- This step focuses on getting to know the dataset you're working with. As mentioned above, this step tends to blend with the _Scrub_ step mentioned above.  During this step, you'll create visualizations to really get a feel for your dataset.  You'll focus on things such as understanding the distribution of different columns, checking for multicollinearity, and other tasks liek that.  If your project is a classification task, you may check the balance of the different classes in your dataset.  If your problem is a regression task, you may check that the dataset meets the assumptions necessary for a regression task.  

- At the end of this step, you should have a dataset ready for modeling that you've thoroughly explored and are extremely familiar with.  

**MODEL**

- This step, as with the last two frameworks, is also pretty self-explanatory. It consists of building and tuning models using all the tools you have in your data science toolbox.  In practice, this often means defining a threshold for success, selecting machine learning algorithms to test on the project, and tuning the ones that show promise to try and increase your results.  As with the other stages, it is both common and accepted to realize something, jump back to a previous stage like _Scrub_ or _Explore_, and make some changes to see how it affects the model.  

**iNTERPRET**

- During this step, you'll interpret the results of your model(s), and communicate results to stakeholders.  As with the other frameworks, communication is incredibily important! During this stage, you may come to realize that further investigation is needed, or more data.  That's totally fine--figure out what's needed, go get it, and start the process over! If your results are satisfactory to all stakeholders involved, you may also go from this stage right into productionizing your model and automating processes necessary to support it.  



<font color='red'>Note: Delete this markdown cell from your final project notebook</font>

## PROCESS CHECKLIST


> Keep in mind that it is normal to jump between the OSEMN phases and some of them will blend together, like SCRUB and EXPLORE.

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

# <font color='red'> START YOUR CODE BELOW:</font>

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

    Requirement already up-to-date: fsds_100719 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (0.4.34)
    Requirement already satisfied, skipping upgrade: seaborn in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.9.0)
    Requirement already satisfied, skipping upgrade: shap in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.32.1)
    Requirement already satisfied, skipping upgrade: pandas in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.25.1)
    Requirement already satisfied, skipping upgrade: missingno in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.4.2)
    Requirement already satisfied, skipping upgrade: IPython in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (7.8.0)
    Requirement already satisfied, skipping upgrade: pandas-profiling in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (2.3.0)
    Requirement already satisfied, skipping upgrade: pprint in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.1)
    Requirement already satisfied, skipping upgrade: ipywidgets in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (7.5.1)
    Requirement already satisfied, skipping upgrade: scipy in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.3.1)
    Requirement already satisfied, skipping upgrade: pyperclip in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.7.0)
    Requirement already satisfied, skipping upgrade: matplotlib in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (3.1.1)
    Requirement already satisfied, skipping upgrade: numpy in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (1.16.5)
    Requirement already satisfied, skipping upgrade: tzlocal in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (2.0.0)
    Requirement already satisfied, skipping upgrade: scikit-learn in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from fsds_100719) (0.21.2)
    Requirement already satisfied, skipping upgrade: tqdm>4.25.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from shap->fsds_100719) (4.36.1)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.6.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas->fsds_100719) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas->fsds_100719) (2019.2)
    Requirement already satisfied, skipping upgrade: pickleshare in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.7.5)
    Requirement already satisfied, skipping upgrade: setuptools>=18.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (41.2.0)
    Requirement already satisfied, skipping upgrade: decorator in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (4.4.0)
    Requirement already satisfied, skipping upgrade: backcall in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.1.0)
    Requirement already satisfied, skipping upgrade: traitlets>=4.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (4.3.2)
    Requirement already satisfied, skipping upgrade: pexpect; sys_platform != "win32" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (4.7.0)
    Requirement already satisfied, skipping upgrade: appnope; sys_platform == "darwin" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.1.0)
    Requirement already satisfied, skipping upgrade: jedi>=0.10 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (0.15.1)
    Requirement already satisfied, skipping upgrade: pygments in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (2.4.2)
    Requirement already satisfied, skipping upgrade: prompt-toolkit<2.1.0,>=2.0.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from IPython->fsds_100719) (2.0.9)
    Requirement already satisfied, skipping upgrade: jinja2>=2.8 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (2.10.1)
    Requirement already satisfied, skipping upgrade: htmlmin>=0.1.12 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (0.1.12)
    Requirement already satisfied, skipping upgrade: confuse>=1.0.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (1.0.0)
    Requirement already satisfied, skipping upgrade: phik>=0.9.8 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (0.9.8)
    Requirement already satisfied, skipping upgrade: astropy in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pandas-profiling->fsds_100719) (3.2.3)
    Requirement already satisfied, skipping upgrade: ipykernel>=4.5.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipywidgets->fsds_100719) (5.1.2)
    Requirement already satisfied, skipping upgrade: widgetsnbextension~=3.5.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipywidgets->fsds_100719) (3.5.1)
    Requirement already satisfied, skipping upgrade: nbformat>=4.2.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipywidgets->fsds_100719) (4.4.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from matplotlib->fsds_100719) (0.10.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from matplotlib->fsds_100719) (1.1.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from matplotlib->fsds_100719) (2.4.2)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from scikit-learn->fsds_100719) (0.13.2)
    Requirement already satisfied, skipping upgrade: six>=1.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from python-dateutil>=2.6.1->pandas->fsds_100719) (1.12.0)
    Requirement already satisfied, skipping upgrade: ipython_genutils in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from traitlets>=4.2->IPython->fsds_100719) (0.2.0)
    Requirement already satisfied, skipping upgrade: ptyprocess>=0.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pexpect; sys_platform != "win32"->IPython->fsds_100719) (0.6.0)
    Requirement already satisfied, skipping upgrade: parso>=0.5.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jedi>=0.10->IPython->fsds_100719) (0.5.1)
    Requirement already satisfied, skipping upgrade: wcwidth in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from prompt-toolkit<2.1.0,>=2.0.0->IPython->fsds_100719) (0.1.7)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jinja2>=2.8->pandas-profiling->fsds_100719) (1.1.1)
    Requirement already satisfied, skipping upgrade: pyyaml in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from confuse>=1.0.0->pandas-profiling->fsds_100719) (5.1.2)
    Requirement already satisfied, skipping upgrade: nbconvert>=5.3.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (5.5.0)
    Requirement already satisfied, skipping upgrade: pytest-pylint>=0.13.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (0.14.1)
    Requirement already satisfied, skipping upgrade: numba>=0.38.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (0.46.0)
    Requirement already satisfied, skipping upgrade: jupyter-client>=5.2.3 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (5.3.3)
    Requirement already satisfied, skipping upgrade: pytest>=4.0.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from phik>=0.9.8->pandas-profiling->fsds_100719) (5.3.0)
    Requirement already satisfied, skipping upgrade: tornado>=4.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from ipykernel>=4.5.1->ipywidgets->fsds_100719) (6.0.3)
    Requirement already satisfied, skipping upgrade: notebook>=4.4.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (5.7.8)
    Requirement already satisfied, skipping upgrade: jsonschema!=2.5.0,>=2.4 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbformat>=4.2.0->ipywidgets->fsds_100719) (3.0.2)
    Requirement already satisfied, skipping upgrade: jupyter_core in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbformat>=4.2.0->ipywidgets->fsds_100719) (4.5.0)
    Requirement already satisfied, skipping upgrade: testpath in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.4.2)
    Requirement already satisfied, skipping upgrade: bleach in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (1.5.0)
    Requirement already satisfied, skipping upgrade: pandocfilters>=1.4.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (1.4.2)
    Requirement already satisfied, skipping upgrade: entrypoints>=0.2.2 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.3)
    Requirement already satisfied, skipping upgrade: defusedxml in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.6.0)
    Requirement already satisfied, skipping upgrade: mistune>=0.8.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.8.4)
    Requirement already satisfied, skipping upgrade: pylint>=1.4.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (2.4.4)
    Requirement already satisfied, skipping upgrade: llvmlite>=0.30.0dev0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from numba>=0.38.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.30.0)
    Requirement already satisfied, skipping upgrade: pyzmq>=13 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jupyter-client>=5.2.3->phik>=0.9.8->pandas-profiling->fsds_100719) (18.1.0)
    Requirement already satisfied, skipping upgrade: py>=1.5.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (1.8.0)
    Requirement already satisfied, skipping upgrade: packaging in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (19.2)
    Requirement already satisfied, skipping upgrade: more-itertools>=4.0.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (7.0.0)
    Requirement already satisfied, skipping upgrade: pluggy<1.0,>=0.12 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (0.12.0)
    Requirement already satisfied, skipping upgrade: attrs>=17.4.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (19.1.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata>=0.12; python_version < "3.8" in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (0.17)
    Requirement already satisfied, skipping upgrade: Send2Trash in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (1.5.0)
    Requirement already satisfied, skipping upgrade: terminado>=0.8.1 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (0.8.2)
    Requirement already satisfied, skipping upgrade: prometheus-client in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->fsds_100719) (0.7.1)
    Requirement already satisfied, skipping upgrade: pyrsistent>=0.14.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->fsds_100719) (0.14.11)
    Requirement already satisfied, skipping upgrade: html5lib!=0.9999,!=0.99999,<0.99999999,>=0.999 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from bleach->nbconvert>=5.3.1->phik>=0.9.8->pandas-profiling->fsds_100719) (0.9999999)
    Requirement already satisfied, skipping upgrade: isort<5,>=4.2.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (4.3.21)
    Requirement already satisfied, skipping upgrade: mccabe<0.7,>=0.6 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (0.6.1)
    Requirement already satisfied, skipping upgrade: astroid<2.4,>=2.3.0 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from pylint>=1.4.5->pytest-pylint>=0.13.0->phik>=0.9.8->pandas-profiling->fsds_100719) (2.3.3)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /anaconda3/envs/learn-env/lib/python3.6/site-packages (from importlib-metadata>=0.12; python_version < "3.8"->pytest>=4.0.2->phik>=0.9.8->pandas-profiling->fsds_100719) (0.5.1)
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
    65.0          1
    2050.0        1
    2350.0        1
    1024.0        1
    295.0         1
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


## Fix bedrooms outlier '33'.


```python
# 33 max value for bedrooms looks weird
# judging by numbers below, the house with 33 rooms is much more comparable to a 3 or 4bd house in sqft  
# and number of bathrooms
df_basic['bedrooms'].unique()
```




    array([ 3,  2,  4,  5,  1,  6,  7,  8,  9, 11, 10, 33])




```python
df_basic.loc[(df_basic['bedrooms'] == 3) | (df_basic['bedrooms'] == 4)]
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
    <tr>
      <td>5</td>
      <td>1230000.0</td>
      <td>4</td>
      <td>4.50</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>2001</td>
      <td>0.0</td>
      <td>98053</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>4760</td>
      <td>101930</td>
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
    </tr>
    <tr>
      <td>21590</td>
      <td>1010000.0</td>
      <td>4</td>
      <td>3.50</td>
      <td>3510</td>
      <td>7200</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5537</td>
      <td>-122.398</td>
      <td>2050</td>
      <td>6200</td>
    </tr>
    <tr>
      <td>21591</td>
      <td>475000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1310</td>
      <td>1294</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98116</td>
      <td>47.5773</td>
      <td>-122.409</td>
      <td>1330</td>
      <td>1265</td>
    </tr>
    <tr>
      <td>21592</td>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1509</td>
    </tr>
    <tr>
      <td>21593</td>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>1830</td>
      <td>7200</td>
    </tr>
    <tr>
      <td>21595</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
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
<p>16580 rows × 16 columns</p>
</div>




```python
df_basic.loc[(df_basic['bedrooms'] == 9) | (df_basic['bedrooms'] == 10)]
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
      <td>4092</td>
      <td>599999.0</td>
      <td>9</td>
      <td>4.50</td>
      <td>3830</td>
      <td>6988</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1938</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6927</td>
      <td>-122.338</td>
      <td>1460</td>
      <td>6291</td>
    </tr>
    <tr>
      <td>4231</td>
      <td>700000.0</td>
      <td>9</td>
      <td>3.00</td>
      <td>3680</td>
      <td>4400</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1908</td>
      <td>0.0</td>
      <td>98102</td>
      <td>47.6374</td>
      <td>-122.324</td>
      <td>1960</td>
      <td>2450</td>
    </tr>
    <tr>
      <td>6073</td>
      <td>1280000.0</td>
      <td>9</td>
      <td>4.50</td>
      <td>3650</td>
      <td>5000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1915</td>
      <td>2010.0</td>
      <td>98105</td>
      <td>47.6604</td>
      <td>-122.289</td>
      <td>2510</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>8537</td>
      <td>450000.0</td>
      <td>9</td>
      <td>7.50</td>
      <td>4050</td>
      <td>6504</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1996</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5923</td>
      <td>-122.301</td>
      <td>1448</td>
      <td>3866</td>
    </tr>
    <tr>
      <td>13301</td>
      <td>1150000.0</td>
      <td>10</td>
      <td>5.25</td>
      <td>4590</td>
      <td>10920</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98004</td>
      <td>47.5861</td>
      <td>-122.113</td>
      <td>2730</td>
      <td>10400</td>
    </tr>
    <tr>
      <td>15147</td>
      <td>650000.0</td>
      <td>10</td>
      <td>2.00</td>
      <td>3610</td>
      <td>11914</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1958</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5705</td>
      <td>-122.175</td>
      <td>2040</td>
      <td>11914</td>
    </tr>
    <tr>
      <td>16830</td>
      <td>1400000.0</td>
      <td>9</td>
      <td>4.00</td>
      <td>4620</td>
      <td>5508</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>1915</td>
      <td>0.0</td>
      <td>98105</td>
      <td>47.6684</td>
      <td>-122.309</td>
      <td>2710</td>
      <td>4320</td>
    </tr>
    <tr>
      <td>18428</td>
      <td>934000.0</td>
      <td>9</td>
      <td>3.00</td>
      <td>2820</td>
      <td>4480</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1918</td>
      <td>0.0</td>
      <td>98105</td>
      <td>47.6654</td>
      <td>-122.307</td>
      <td>2460</td>
      <td>4400</td>
    </tr>
    <tr>
      <td>19239</td>
      <td>660000.0</td>
      <td>10</td>
      <td>3.00</td>
      <td>2920</td>
      <td>3745</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1913</td>
      <td>0.0</td>
      <td>98105</td>
      <td>47.6635</td>
      <td>-122.320</td>
      <td>1810</td>
      <td>3745</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 33 max value for bedrooms looks weird
df_basic.loc[(df_basic['bedrooms'] == 33) | (df_basic['bedrooms'] == 11)]

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
      <td>8748</td>
      <td>520000.0</td>
      <td>11</td>
      <td>3.00</td>
      <td>3000</td>
      <td>4960</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1918</td>
      <td>1999.0</td>
      <td>98106</td>
      <td>47.5560</td>
      <td>-122.363</td>
      <td>1420</td>
      <td>4960</td>
    </tr>
    <tr>
      <td>15856</td>
      <td>640000.0</td>
      <td>33</td>
      <td>1.75</td>
      <td>1620</td>
      <td>6000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1947</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6878</td>
      <td>-122.331</td>
      <td>1330</td>
      <td>4700</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_basic.loc[(df_basic['bedrooms'] == 33), 'bedrooms'] = 3
```


```python
df_basic.loc[(df_basic['bedrooms'] == 33)]
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
  </tbody>
</table>
</div>




```python
# this looks much better
df_basic.plot(kind='scatter', x='bedrooms', y='price', alpha=0.4, color='b');
```


![png](output_59_0.png)


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


![png](output_62_0.png)



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


![png](output_64_0.png)



```python
#checking plots for categorical columns
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(df_basic)[9:13], axes):
    df_basic.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')
    
plt.tight_layout()
```


![png](output_65_0.png)



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


![png](output_67_0.png)



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


```python
df_basic['zipcode'].nunique()
```




    70



## Check for multicollinearity.

### Create heatmap for collinearity check. 


```python
fig, axes = plt.subplots(figsize=(13,13))
sns.heatmap(df_basic.corr().round(3), center=0, square=True, ax=axes, annot=True);
axes.set_ylim(len(df_basic.corr()),-0.5,+0.5)
```




    (16, -0.5)




![png](output_89_1.png)


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


![png](output_97_0.png)



```python
# normalization thoughts:
# the only continuous variables I have are sqft living, sqft lot, living 15, lot 15. 
# The rest are basically categorical. Not sure if those need to be normalized. 
```

## Data transformations 1 - Z-scoring.

### Normalize data by filtering outliers with z-scores.


```python
# outliers to filter: 
#     bedroom 
#     bathroom, 
#     sqft living, 
#     sqft lot, 
#     floors, 
#     grade, 
#     lat, 
#     long, 
#     sqft living 15, 
#     sqft lot 15
```

### Filter bedrooms.


```python
# outliers to filter - bedroom
df_basic.plot(kind='scatter', x='bedrooms', y='price', alpha=0.4, color='b');
```


![png](output_103_0.png)



```python
# bedroom zscore filter1
import scipy.stats as stats

z_score_bed = np.abs(stats.zscore(df_basic['bedrooms']))

df_basic['z_score_bedrooms'] = z_score_bed

df_filtered1 = df_basic.loc[df_basic['z_score_bedrooms'] < 3]

display(df_filtered1.head())
display(sns.boxplot(df_filtered1['bedrooms']))
display(df_filtered1.plot(kind='scatter', x='bedrooms', y='price', alpha=0.4, color='b'))
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
      <th>z_score_bedrooms</th>
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
      <td>0.412580</td>
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
      <td>0.412580</td>
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
      <td>1.520031</td>
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
      <td>0.694872</td>
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
      <td>0.412580</td>
    </tr>
  </tbody>
</table>
</div>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c22c2f198>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c22bc8470>



![png](output_104_3.png)



![png](output_104_4.png)



```python
# so it looks like my bedrooms are filtered from 1 to 6 bedrooms now
```

### Filter bathrooms.


```python
# outliers to filter - bathrooms
df_basic.plot(kind='scatter', x='bathrooms', y='price', alpha=0.4, color='b');
```


![png](output_107_0.png)



```python
z_score_bath = np.abs(stats.zscore(df_basic['bathrooms']))

df_basic['z_score_bathrooms'] = z_score_bath

df_filtered1 = df_basic.loc[df_basic['z_score_bathrooms'] < 3]

display(df_filtered1.head())
display(sns.boxplot(df_filtered1['bathrooms']))
display(df_filtered1.plot(kind='scatter', x='bathrooms', y='price', alpha=0.4, color='b'))
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
      <th>z_score_bedrooms</th>
      <th>z_score_bathrooms</th>
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
      <td>0.412580</td>
      <td>1.454958</td>
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
      <td>0.412580</td>
      <td>0.171160</td>
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
      <td>1.520031</td>
      <td>1.454958</td>
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
      <td>0.694872</td>
      <td>1.146831</td>
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
      <td>0.412580</td>
      <td>0.154064</td>
    </tr>
  </tbody>
</table>
</div>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c233c6940>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c1dc95da0>



![png](output_108_3.png)



![png](output_108_4.png)



```python
# it looks like the z score filter put my bathroom range from .5 to 4.5. sweet. 
```

### Filter sqft living.


```python
# outliers to filter - sqft living
df_basic.plot(kind='scatter', x='sqft_living', y='price', alpha=0.4, color='b');
```


![png](output_111_0.png)



```python
z_score_sqft_living = np.abs(stats.zscore(df_basic['sqft_living']))

df_basic['z_score_sqft_living'] = z_score_sqft_living

df_filtered1 = df_basic.loc[df_basic['z_score_sqft_living'] < 3]

display(df_filtered1.head())
display(sns.boxplot(df_filtered1['sqft_living']))
display(df_filtered1.plot(kind='scatter', x='sqft_living', y='price', alpha=0.4, color='b'))
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
      <th>z_score_bedrooms</th>
      <th>z_score_bathrooms</th>
      <th>z_score_sqft_living</th>
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
      <td>0.412580</td>
      <td>1.454958</td>
      <td>0.982962</td>
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
      <td>0.412580</td>
      <td>0.171160</td>
      <td>0.529902</td>
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
      <td>1.520031</td>
      <td>1.454958</td>
      <td>1.429203</td>
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
      <td>0.694872</td>
      <td>1.146831</td>
      <td>0.134017</td>
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
      <td>0.412580</td>
      <td>0.154064</td>
      <td>0.438766</td>
    </tr>
  </tbody>
</table>
</div>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c233e9128>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c23319f28>



![png](output_112_3.png)



![png](output_112_4.png)



```python
#it looks like the zscore filtered my sqft living from about 500 to 4800
```

### Filter sqft lot.


```python
# outliers to filter - sqft lot
df_basic.plot(kind='scatter', x='sqft_lot', y='price', alpha=0.4, color='b');
```


![png](output_115_0.png)



```python
z_score_sqft_lot = np.abs(stats.zscore(df_basic['sqft_lot']))

df_basic['z_score_sqft_lot'] = z_score_sqft_lot

df_filtered1 = df_basic.loc[df_basic['z_score_sqft_lot'] < 3]

display(df_filtered1.head())
display(sns.boxplot(df_filtered1['sqft_lot']))
display(df_filtered1.plot(kind='scatter', x='sqft_lot', y='price', alpha=0.4, color='b'))
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
      <th>z_score_bedrooms</th>
      <th>z_score_bathrooms</th>
      <th>z_score_sqft_living</th>
      <th>z_score_sqft_lot</th>
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
      <td>0.412580</td>
      <td>1.454958</td>
      <td>0.982962</td>
      <td>0.228222</td>
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
      <td>0.412580</td>
      <td>0.171160</td>
      <td>0.529902</td>
      <td>0.189889</td>
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
      <td>1.520031</td>
      <td>1.454958</td>
      <td>1.429203</td>
      <td>0.123478</td>
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
      <td>0.694872</td>
      <td>1.146831</td>
      <td>0.134017</td>
      <td>0.243874</td>
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
      <td>0.412580</td>
      <td>0.154064</td>
      <td>0.438766</td>
      <td>0.169710</td>
    </tr>
  </tbody>
</table>
</div>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c23304cf8>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c29590a90>



![png](output_116_3.png)



![png](output_116_4.png)



```python
# it looks like the zscore filtered my sqft lot from 0 to 140000, way better than the outlier 1.6 million
```

### Filter sqft lot 15.


```python
# outliers to filter - sqft lot
df_basic.plot(kind='scatter', x='sqft_lot', y='price', alpha=0.4, color='b');
```


![png](output_119_0.png)



```python
z_score_sqft_lot15 = np.abs(stats.zscore(df_basic['sqft_lot15']))

df_basic['z_score_sqft_lot15'] = z_score_sqft_lot15

df_filtered1 = df_basic.loc[df_basic['z_score_sqft_lot15'] < 3]

display(df_filtered1.head())
display(sns.boxplot(df_filtered1['sqft_lot15']))
display(df_filtered1.plot(kind='scatter', x='sqft_lot15', y='price', alpha=0.4, color='b'))
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
      <th>z_score_bedrooms</th>
      <th>z_score_bathrooms</th>
      <th>z_score_sqft_living</th>
      <th>z_score_sqft_lot</th>
      <th>z_score_sqft_lot15</th>
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
      <td>0.412580</td>
      <td>1.454958</td>
      <td>0.982962</td>
      <td>0.228222</td>
      <td>0.260586</td>
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
      <td>0.412580</td>
      <td>0.171160</td>
      <td>0.529902</td>
      <td>0.189889</td>
      <td>0.187849</td>
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
      <td>1.520031</td>
      <td>1.454958</td>
      <td>1.429203</td>
      <td>0.123478</td>
      <td>0.172380</td>
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
      <td>0.694872</td>
      <td>1.146831</td>
      <td>0.134017</td>
      <td>0.243874</td>
      <td>0.284356</td>
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
      <td>0.412580</td>
      <td>0.154064</td>
      <td>0.438766</td>
      <td>0.169710</td>
      <td>0.192822</td>
    </tr>
  </tbody>
</table>
</div>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c205a6278>



    <matplotlib.axes._subplots.AxesSubplot at 0x1c26286da0>



![png](output_120_3.png)



![png](output_120_4.png)



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
      <th>z_score_bedrooms</th>
      <th>z_score_bathrooms</th>
      <th>z_score_sqft_living</th>
      <th>z_score_sqft_lot</th>
      <th>z_score_sqft_lot15</th>
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
      <td>0.412580</td>
      <td>1.454958</td>
      <td>0.982962</td>
      <td>0.228222</td>
      <td>0.260586</td>
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
      <td>0.412580</td>
      <td>0.171160</td>
      <td>0.529902</td>
      <td>0.189889</td>
      <td>0.187849</td>
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
      <td>1.520031</td>
      <td>1.454958</td>
      <td>1.429203</td>
      <td>0.123478</td>
      <td>0.172380</td>
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
      <td>0.694872</td>
      <td>1.146831</td>
      <td>0.134017</td>
      <td>0.243874</td>
      <td>0.284356</td>
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
      <td>0.412580</td>
      <td>0.154064</td>
      <td>0.438766</td>
      <td>0.169710</td>
      <td>0.192822</td>
    </tr>
  </tbody>
</table>
</div>



### Make filtered1 into copy to preserve basic. 


```python
df_filtered1 = df_filtered1.copy()
```


```python
df_basic.shape
```




    (21420, 20)




```python
df_filtered1.shape
```




    (21059, 20)



### Drop unwanted zscore cols.


```python
df_filtered1.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated',
           'zipcode', 'lat', 'long', 'sqft_lot15', 'z_score_bedrooms',
           'z_score_bathrooms', 'z_score_sqft_living', 'z_score_sqft_lot',
           'z_score_sqft_lot15'],
          dtype='object')




```python
df_filtered1.drop(['z_score_bedrooms',
       'z_score_bathrooms', 'z_score_sqft_living', 'z_score_sqft_lot',
       'z_score_sqft_lot15'], axis=1, inplace=True)
```


```python
df_filtered1.head()
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



### Check model normality after zscore filter outliers. 


```python
df_filtered1.hist(figsize=(9,9));
```


![png](output_131_0.png)


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
model = ols(formula=formula, data=df_filtered1).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.692</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.691</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   3370.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 19 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>18:52:37</td>     <th>  Log-Likelihood:    </th> <td>-2.8738e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21059</td>      <th>  AIC:               </th>  <td>5.748e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21044</td>      <th>  BIC:               </th>  <td>5.749e+05</td> 
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
  <th>Intercept</th>    <td> 9.775e+06</td> <td> 2.91e+06</td> <td>    3.354</td> <td> 0.001</td> <td> 4.06e+06</td> <td> 1.55e+07</td>
</tr>
<tr>
  <th>bedrooms</th>     <td>-4.528e+04</td> <td> 2036.798</td> <td>  -22.232</td> <td> 0.000</td> <td>-4.93e+04</td> <td>-4.13e+04</td>
</tr>
<tr>
  <th>bathrooms</th>    <td> 4.098e+04</td> <td> 3320.377</td> <td>   12.341</td> <td> 0.000</td> <td> 3.45e+04</td> <td> 4.75e+04</td>
</tr>
<tr>
  <th>sqft_living</th>  <td>  197.5426</td> <td>    3.243</td> <td>   60.911</td> <td> 0.000</td> <td>  191.186</td> <td>  203.899</td>
</tr>
<tr>
  <th>sqft_lot</th>     <td>    0.0700</td> <td>    0.063</td> <td>    1.106</td> <td> 0.269</td> <td>   -0.054</td> <td>    0.194</td>
</tr>
<tr>
  <th>floors</th>       <td> 9257.4954</td> <td> 3337.068</td> <td>    2.774</td> <td> 0.006</td> <td> 2716.586</td> <td> 1.58e+04</td>
</tr>
<tr>
  <th>waterfront</th>   <td> 7.974e+05</td> <td> 1.75e+04</td> <td>   45.696</td> <td> 0.000</td> <td> 7.63e+05</td> <td> 8.32e+05</td>
</tr>
<tr>
  <th>condition</th>    <td> 2.655e+04</td> <td> 2406.896</td> <td>   11.029</td> <td> 0.000</td> <td> 2.18e+04</td> <td> 3.13e+04</td>
</tr>
<tr>
  <th>grade</th>        <td> 1.096e+05</td> <td> 2097.631</td> <td>   52.230</td> <td> 0.000</td> <td> 1.05e+05</td> <td> 1.14e+05</td>
</tr>
<tr>
  <th>yr_built</th>     <td>-2865.4606</td> <td>   73.533</td> <td>  -38.968</td> <td> 0.000</td> <td>-3009.591</td> <td>-2721.331</td>
</tr>
<tr>
  <th>yr_renovated</th> <td>   24.8156</td> <td>    4.079</td> <td>    6.084</td> <td> 0.000</td> <td>   16.821</td> <td>   32.810</td>
</tr>
<tr>
  <th>zipcode</th>      <td> -558.2711</td> <td>   33.551</td> <td>  -16.639</td> <td> 0.000</td> <td> -624.034</td> <td> -492.508</td>
</tr>
<tr>
  <th>lat</th>          <td> 5.802e+05</td> <td>  1.1e+04</td> <td>   52.783</td> <td> 0.000</td> <td> 5.59e+05</td> <td> 6.02e+05</td>
</tr>
<tr>
  <th>long</th>         <td>-1.824e+05</td> <td> 1.34e+04</td> <td>  -13.562</td> <td> 0.000</td> <td>-2.09e+05</td> <td>-1.56e+05</td>
</tr>
<tr>
  <th>sqft_lot15</th>   <td>   -1.3836</td> <td>    0.179</td> <td>   -7.727</td> <td> 0.000</td> <td>   -1.735</td> <td>   -1.033</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>17243.944</td> <th>  Durbin-Watson:     </th>  <td>   2.001</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1462129.457</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.406</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>43.248</td>   <th>  Cond. No.          </th>  <td>2.06e+08</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.06e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# You'll notice that our R2 value is 0.692 which is eh.

# The P value for sqft_lot is at 0.269, well above the accepted 0.05 so that may have some issues with collinearity. 
# We can check the heat map on that one again. 

# Our skew is 3.4.
# Kurtosis is 43.2.
# Condition is 2.06^8
```

### Create model using numerical and categorical predictors.

In order to be sure our categorical variables are not being used in regression as values instead of labels, we will use stats to use a C() operator which will essentially one hot those specific variables


```python
# create model using list of predictors by typing them all out. 
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'C(waterfront)', 'C(condition)', 'grade', 'C(yr_built)', 'C(yr_renovated)',
       'C(zipcode)', 'lat', 'long', 'sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = smf.ols(formula=formula, data=df_filtered1).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.804</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.801</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   318.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 19 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>18:52:43</td>     <th>  Log-Likelihood:    </th> <td>-2.8262e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21059</td>      <th>  AIC:               </th>  <td>5.658e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20791</td>      <th>  BIC:               </th>  <td>5.679e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>   267</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>-3.136e+07</td> <td> 6.64e+06</td> <td>   -4.720</td> <td> 0.000</td> <td>-4.44e+07</td> <td>-1.83e+07</td>
</tr>
<tr>
  <th>C(waterfront)[T.1.0]</th>      <td> 8.542e+05</td> <td> 1.45e+04</td> <td>   59.037</td> <td> 0.000</td> <td> 8.26e+05</td> <td> 8.83e+05</td>
</tr>
<tr>
  <th>C(condition)[T.2]</th>         <td> 5.493e+04</td> <td>  3.4e+04</td> <td>    1.616</td> <td> 0.106</td> <td>-1.17e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>C(condition)[T.3]</th>         <td> 4.746e+04</td> <td> 3.14e+04</td> <td>    1.510</td> <td> 0.131</td> <td>-1.41e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>C(condition)[T.4]</th>         <td> 6.917e+04</td> <td> 3.14e+04</td> <td>    2.200</td> <td> 0.028</td> <td> 7547.236</td> <td> 1.31e+05</td>
</tr>
<tr>
  <th>C(condition)[T.5]</th>         <td> 1.031e+05</td> <td> 3.16e+04</td> <td>    3.262</td> <td> 0.001</td> <td> 4.12e+04</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1901]</th>       <td>-1.117e+05</td> <td> 3.54e+04</td> <td>   -3.153</td> <td> 0.002</td> <td>-1.81e+05</td> <td>-4.23e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1902]</th>       <td>-6.575e+04</td> <td> 3.65e+04</td> <td>   -1.800</td> <td> 0.072</td> <td>-1.37e+05</td> <td> 5861.110</td>
</tr>
<tr>
  <th>C(yr_built)[T.1903]</th>       <td>-4.726e+04</td> <td> 3.04e+04</td> <td>   -1.555</td> <td> 0.120</td> <td>-1.07e+05</td> <td> 1.23e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1904]</th>       <td>-1.861e+04</td> <td> 3.07e+04</td> <td>   -0.607</td> <td> 0.544</td> <td>-7.87e+04</td> <td> 4.15e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1905]</th>       <td> 1.254e+04</td> <td> 2.64e+04</td> <td>    0.474</td> <td> 0.635</td> <td>-3.93e+04</td> <td> 6.43e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1906]</th>       <td>-2.187e+04</td> <td> 2.49e+04</td> <td>   -0.878</td> <td> 0.380</td> <td>-7.07e+04</td> <td> 2.69e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1907]</th>       <td>-7453.1960</td> <td> 2.72e+04</td> <td>   -0.274</td> <td> 0.784</td> <td>-6.09e+04</td> <td> 4.59e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1908]</th>       <td>-4.597e+04</td> <td> 2.53e+04</td> <td>   -1.815</td> <td> 0.069</td> <td>-9.56e+04</td> <td> 3667.808</td>
</tr>
<tr>
  <th>C(yr_built)[T.1909]</th>       <td>-1813.8779</td> <td> 2.48e+04</td> <td>   -0.073</td> <td> 0.942</td> <td>-5.05e+04</td> <td> 4.68e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1910]</th>       <td> 1.118e+04</td> <td> 2.31e+04</td> <td>    0.484</td> <td> 0.628</td> <td>-3.41e+04</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1911]</th>       <td>-3555.5462</td> <td> 2.66e+04</td> <td>   -0.134</td> <td> 0.894</td> <td>-5.58e+04</td> <td> 4.86e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1912]</th>       <td>-1.461e+04</td> <td> 2.62e+04</td> <td>   -0.558</td> <td> 0.577</td> <td>-6.59e+04</td> <td> 3.67e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1913]</th>       <td>-8219.8580</td> <td> 2.86e+04</td> <td>   -0.287</td> <td> 0.774</td> <td>-6.44e+04</td> <td> 4.79e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1914]</th>       <td>-1.834e+04</td> <td> 2.89e+04</td> <td>   -0.634</td> <td> 0.526</td> <td>-7.51e+04</td> <td> 3.84e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1915]</th>       <td>-1.866e+04</td> <td> 2.76e+04</td> <td>   -0.675</td> <td> 0.500</td> <td>-7.28e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1916]</th>       <td> 6124.1018</td> <td> 2.59e+04</td> <td>    0.237</td> <td> 0.813</td> <td>-4.45e+04</td> <td> 5.68e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1917]</th>       <td>-2.268e+04</td> <td> 2.84e+04</td> <td>   -0.798</td> <td> 0.425</td> <td>-7.84e+04</td> <td>  3.3e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1918]</th>       <td>-2.217e+04</td> <td> 2.36e+04</td> <td>   -0.940</td> <td> 0.347</td> <td>-6.84e+04</td> <td> 2.41e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1919]</th>       <td>-2.643e+04</td> <td> 2.54e+04</td> <td>   -1.042</td> <td> 0.297</td> <td>-7.62e+04</td> <td> 2.33e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1920]</th>       <td> -2.18e+04</td> <td> 2.46e+04</td> <td>   -0.885</td> <td> 0.376</td> <td>-7.01e+04</td> <td> 2.65e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1921]</th>       <td>-1.642e+04</td> <td> 2.65e+04</td> <td>   -0.620</td> <td> 0.535</td> <td>-6.83e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1922]</th>       <td>-5156.4625</td> <td>  2.5e+04</td> <td>   -0.207</td> <td> 0.836</td> <td>-5.41e+04</td> <td> 4.38e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1923]</th>       <td>-9454.6436</td> <td> 2.57e+04</td> <td>   -0.369</td> <td> 0.712</td> <td>-5.97e+04</td> <td> 4.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1924]</th>       <td>-2.055e+04</td> <td> 2.29e+04</td> <td>   -0.898</td> <td> 0.369</td> <td>-6.54e+04</td> <td> 2.43e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1925]</th>       <td>-3.809e+04</td> <td> 2.22e+04</td> <td>   -1.714</td> <td> 0.087</td> <td>-8.17e+04</td> <td> 5479.639</td>
</tr>
<tr>
  <th>C(yr_built)[T.1926]</th>       <td>-8127.6901</td> <td> 2.18e+04</td> <td>   -0.372</td> <td> 0.710</td> <td>-5.09e+04</td> <td> 3.47e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1927]</th>       <td>-2.573e+04</td> <td> 2.38e+04</td> <td>   -1.080</td> <td> 0.280</td> <td>-7.24e+04</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1928]</th>       <td> 5767.8989</td> <td> 2.33e+04</td> <td>    0.247</td> <td> 0.805</td> <td>   -4e+04</td> <td> 5.15e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1929]</th>       <td>-5.626e+04</td> <td> 2.38e+04</td> <td>   -2.363</td> <td> 0.018</td> <td>-1.03e+05</td> <td>-9593.008</td>
</tr>
<tr>
  <th>C(yr_built)[T.1930]</th>       <td>-2.468e+04</td> <td>  2.5e+04</td> <td>   -0.985</td> <td> 0.324</td> <td>-7.38e+04</td> <td> 2.44e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1931]</th>       <td> -3.38e+04</td> <td> 2.79e+04</td> <td>   -1.211</td> <td> 0.226</td> <td>-8.85e+04</td> <td> 2.09e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1932]</th>       <td>-4259.9025</td> <td> 3.38e+04</td> <td>   -0.126</td> <td> 0.900</td> <td>-7.04e+04</td> <td> 6.19e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1933]</th>       <td> 9.721e+04</td> <td> 3.67e+04</td> <td>    2.645</td> <td> 0.008</td> <td> 2.52e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1934]</th>       <td> 9180.6924</td> <td> 4.24e+04</td> <td>    0.216</td> <td> 0.829</td> <td> -7.4e+04</td> <td> 9.24e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1935]</th>       <td> 1.262e+04</td> <td> 3.94e+04</td> <td>    0.320</td> <td> 0.749</td> <td>-6.46e+04</td> <td> 8.98e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1936]</th>       <td> 5.333e+04</td> <td>  3.3e+04</td> <td>    1.618</td> <td> 0.106</td> <td>-1.13e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1937]</th>       <td>-8438.2796</td> <td> 2.74e+04</td> <td>   -0.308</td> <td> 0.758</td> <td>-6.21e+04</td> <td> 4.52e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1938]</th>       <td>-1928.1473</td> <td> 2.95e+04</td> <td>   -0.065</td> <td> 0.948</td> <td>-5.98e+04</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1939]</th>       <td> 4206.6894</td> <td> 2.44e+04</td> <td>    0.172</td> <td> 0.863</td> <td>-4.36e+04</td> <td> 5.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1940]</th>       <td>-1.219e+04</td> <td> 2.26e+04</td> <td>   -0.540</td> <td> 0.589</td> <td>-5.65e+04</td> <td> 3.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1941]</th>       <td> 7778.1663</td> <td> 2.23e+04</td> <td>    0.348</td> <td> 0.728</td> <td> -3.6e+04</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1942]</th>       <td> -1.48e+04</td> <td> 2.14e+04</td> <td>   -0.693</td> <td> 0.488</td> <td>-5.67e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1943]</th>       <td>-1.697e+04</td> <td> 2.23e+04</td> <td>   -0.762</td> <td> 0.446</td> <td>-6.06e+04</td> <td> 2.67e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1944]</th>       <td>-1.312e+04</td> <td> 2.31e+04</td> <td>   -0.567</td> <td> 0.571</td> <td>-5.84e+04</td> <td> 3.22e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1945]</th>       <td>-4677.9948</td> <td> 2.49e+04</td> <td>   -0.188</td> <td> 0.851</td> <td>-5.36e+04</td> <td> 4.42e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1946]</th>       <td>-4.724e+04</td> <td> 2.35e+04</td> <td>   -2.014</td> <td> 0.044</td> <td>-9.32e+04</td> <td>-1254.683</td>
</tr>
<tr>
  <th>C(yr_built)[T.1947]</th>       <td>-4.661e+04</td> <td> 2.09e+04</td> <td>   -2.231</td> <td> 0.026</td> <td>-8.76e+04</td> <td>-5660.071</td>
</tr>
<tr>
  <th>C(yr_built)[T.1948]</th>       <td>-4.518e+04</td> <td> 2.12e+04</td> <td>   -2.131</td> <td> 0.033</td> <td>-8.67e+04</td> <td>-3632.491</td>
</tr>
<tr>
  <th>C(yr_built)[T.1949]</th>       <td>-1.812e+04</td> <td> 2.18e+04</td> <td>   -0.832</td> <td> 0.406</td> <td>-6.08e+04</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1950]</th>       <td>-3.551e+04</td> <td> 2.11e+04</td> <td>   -1.687</td> <td> 0.092</td> <td>-7.68e+04</td> <td> 5756.453</td>
</tr>
<tr>
  <th>C(yr_built)[T.1951]</th>       <td>-5.029e+04</td> <td> 2.12e+04</td> <td>   -2.372</td> <td> 0.018</td> <td>-9.18e+04</td> <td>-8730.466</td>
</tr>
<tr>
  <th>C(yr_built)[T.1952]</th>       <td>-7.093e+04</td> <td> 2.14e+04</td> <td>   -3.318</td> <td> 0.001</td> <td>-1.13e+05</td> <td> -2.9e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1953]</th>       <td>-8.853e+04</td> <td> 2.14e+04</td> <td>   -4.128</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-4.65e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1954]</th>       <td> -7.25e+04</td> <td> 2.06e+04</td> <td>   -3.524</td> <td> 0.000</td> <td>-1.13e+05</td> <td>-3.22e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1955]</th>       <td>-9.017e+04</td> <td> 2.09e+04</td> <td>   -4.318</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-4.92e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1956]</th>       <td>-7.487e+04</td> <td> 2.18e+04</td> <td>   -3.431</td> <td> 0.001</td> <td>-1.18e+05</td> <td>-3.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1957]</th>       <td>-7.779e+04</td> <td> 2.17e+04</td> <td>   -3.583</td> <td> 0.000</td> <td> -1.2e+05</td> <td>-3.52e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1958]</th>       <td>-7.085e+04</td> <td> 2.14e+04</td> <td>   -3.314</td> <td> 0.001</td> <td>-1.13e+05</td> <td>-2.89e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1959]</th>       <td>-7.913e+04</td> <td> 2.04e+04</td> <td>   -3.877</td> <td> 0.000</td> <td>-1.19e+05</td> <td>-3.91e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1960]</th>       <td>-9.054e+04</td> <td> 2.11e+04</td> <td>   -4.284</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-4.91e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1961]</th>       <td>-8.844e+04</td> <td> 2.13e+04</td> <td>   -4.145</td> <td> 0.000</td> <td> -1.3e+05</td> <td>-4.66e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1962]</th>       <td>-9.185e+04</td> <td> 2.06e+04</td> <td>   -4.464</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-5.15e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1963]</th>       <td>-9.226e+04</td> <td> 2.11e+04</td> <td>   -4.383</td> <td> 0.000</td> <td>-1.34e+05</td> <td> -5.1e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1964]</th>       <td>-1.179e+05</td> <td> 2.23e+04</td> <td>   -5.294</td> <td> 0.000</td> <td>-1.62e+05</td> <td>-7.43e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1965]</th>       <td>-1.212e+05</td> <td> 2.19e+04</td> <td>   -5.534</td> <td> 0.000</td> <td>-1.64e+05</td> <td>-7.83e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1966]</th>       <td>-1.184e+05</td> <td> 2.12e+04</td> <td>   -5.590</td> <td> 0.000</td> <td> -1.6e+05</td> <td>-7.69e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1967]</th>       <td>-9.218e+04</td> <td> 2.04e+04</td> <td>   -4.521</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-5.22e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1968]</th>       <td>-8.354e+04</td> <td> 2.02e+04</td> <td>   -4.138</td> <td> 0.000</td> <td>-1.23e+05</td> <td> -4.4e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1969]</th>       <td>-5.168e+04</td> <td> 2.09e+04</td> <td>   -2.478</td> <td> 0.013</td> <td>-9.26e+04</td> <td>-1.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1970]</th>       <td>-8.579e+04</td> <td> 2.33e+04</td> <td>   -3.677</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-4.01e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1971]</th>       <td>-7.506e+04</td> <td> 2.46e+04</td> <td>   -3.046</td> <td> 0.002</td> <td>-1.23e+05</td> <td>-2.68e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1972]</th>       <td>-1.078e+05</td> <td>  2.3e+04</td> <td>   -4.697</td> <td> 0.000</td> <td>-1.53e+05</td> <td>-6.28e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1973]</th>       <td>-9.692e+04</td> <td> 2.29e+04</td> <td>   -4.236</td> <td> 0.000</td> <td>-1.42e+05</td> <td>-5.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1974]</th>       <td>-1.184e+05</td> <td> 2.25e+04</td> <td>   -5.265</td> <td> 0.000</td> <td>-1.63e+05</td> <td>-7.44e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1975]</th>       <td>-1.031e+05</td> <td>  2.2e+04</td> <td>   -4.690</td> <td> 0.000</td> <td>-1.46e+05</td> <td>   -6e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1976]</th>       <td>-1.134e+05</td> <td> 2.11e+04</td> <td>   -5.369</td> <td> 0.000</td> <td>-1.55e+05</td> <td> -7.2e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1977]</th>       <td>-1.202e+05</td> <td> 2.01e+04</td> <td>   -5.988</td> <td> 0.000</td> <td> -1.6e+05</td> <td>-8.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1978]</th>       <td>-1.301e+05</td> <td> 2.02e+04</td> <td>   -6.435</td> <td> 0.000</td> <td> -1.7e+05</td> <td>-9.05e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1979]</th>       <td>-1.404e+05</td> <td> 2.05e+04</td> <td>   -6.864</td> <td> 0.000</td> <td> -1.8e+05</td> <td>   -1e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1980]</th>       <td>-9.825e+04</td> <td> 2.14e+04</td> <td>   -4.600</td> <td> 0.000</td> <td> -1.4e+05</td> <td>-5.64e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1981]</th>       <td>-7.246e+04</td> <td> 2.19e+04</td> <td>   -3.308</td> <td> 0.001</td> <td>-1.15e+05</td> <td>-2.95e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1982]</th>       <td>-7.788e+04</td> <td> 2.48e+04</td> <td>   -3.146</td> <td> 0.002</td> <td>-1.26e+05</td> <td>-2.94e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1983]</th>       <td>-7.506e+04</td> <td> 2.17e+04</td> <td>   -3.462</td> <td> 0.001</td> <td>-1.18e+05</td> <td>-3.26e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1984]</th>       <td>-1.078e+05</td> <td> 2.15e+04</td> <td>   -5.020</td> <td> 0.000</td> <td> -1.5e+05</td> <td>-6.57e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1985]</th>       <td>-9.765e+04</td> <td> 2.15e+04</td> <td>   -4.537</td> <td> 0.000</td> <td> -1.4e+05</td> <td>-5.55e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1986]</th>       <td>-1.122e+05</td> <td> 2.16e+04</td> <td>   -5.185</td> <td> 0.000</td> <td>-1.55e+05</td> <td>-6.98e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1987]</th>       <td>-1.249e+05</td> <td> 2.09e+04</td> <td>   -5.977</td> <td> 0.000</td> <td>-1.66e+05</td> <td>-8.39e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1988]</th>       <td>-1.195e+05</td> <td> 2.11e+04</td> <td>   -5.651</td> <td> 0.000</td> <td>-1.61e+05</td> <td> -7.8e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1989]</th>       <td>-1.075e+05</td> <td>  2.1e+04</td> <td>   -5.108</td> <td> 0.000</td> <td>-1.49e+05</td> <td>-6.62e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1990]</th>       <td>-1.453e+05</td> <td> 2.08e+04</td> <td>   -6.971</td> <td> 0.000</td> <td>-1.86e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1991]</th>       <td>-1.035e+05</td> <td> 2.18e+04</td> <td>   -4.751</td> <td> 0.000</td> <td>-1.46e+05</td> <td>-6.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1992]</th>       <td>-1.213e+05</td> <td> 2.21e+04</td> <td>   -5.495</td> <td> 0.000</td> <td>-1.65e+05</td> <td> -7.8e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1993]</th>       <td>-1.088e+05</td> <td>  2.2e+04</td> <td>   -4.938</td> <td> 0.000</td> <td>-1.52e+05</td> <td>-6.56e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1994]</th>       <td>-1.081e+05</td> <td> 2.14e+04</td> <td>   -5.056</td> <td> 0.000</td> <td> -1.5e+05</td> <td>-6.62e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1995]</th>       <td> -1.03e+05</td> <td> 2.26e+04</td> <td>   -4.558</td> <td> 0.000</td> <td>-1.47e+05</td> <td>-5.87e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1996]</th>       <td>-1.007e+05</td> <td> 2.21e+04</td> <td>   -4.548</td> <td> 0.000</td> <td>-1.44e+05</td> <td>-5.73e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1997]</th>       <td>-1.188e+05</td> <td> 2.25e+04</td> <td>   -5.274</td> <td> 0.000</td> <td>-1.63e+05</td> <td>-7.46e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1998]</th>       <td>-1.227e+05</td> <td> 2.16e+04</td> <td>   -5.693</td> <td> 0.000</td> <td>-1.65e+05</td> <td>-8.05e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1999]</th>       <td>  -8.7e+04</td> <td> 2.13e+04</td> <td>   -4.082</td> <td> 0.000</td> <td>-1.29e+05</td> <td>-4.52e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2000]</th>       <td>-1.042e+05</td> <td> 2.18e+04</td> <td>   -4.773</td> <td> 0.000</td> <td>-1.47e+05</td> <td>-6.14e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2001]</th>       <td>-9.635e+04</td> <td>  2.1e+04</td> <td>   -4.598</td> <td> 0.000</td> <td>-1.37e+05</td> <td>-5.53e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2002]</th>       <td>-1.196e+05</td> <td> 2.17e+04</td> <td>   -5.504</td> <td> 0.000</td> <td>-1.62e+05</td> <td> -7.7e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2003]</th>       <td>-1.051e+05</td> <td> 2.03e+04</td> <td>   -5.179</td> <td> 0.000</td> <td>-1.45e+05</td> <td>-6.53e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2004]</th>       <td>-1.134e+05</td> <td> 2.03e+04</td> <td>   -5.586</td> <td> 0.000</td> <td>-1.53e+05</td> <td>-7.36e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2005]</th>       <td> -1.27e+05</td> <td> 2.02e+04</td> <td>   -6.294</td> <td> 0.000</td> <td>-1.66e+05</td> <td>-8.74e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2006]</th>       <td>-1.291e+05</td> <td> 2.01e+04</td> <td>   -6.409</td> <td> 0.000</td> <td>-1.69e+05</td> <td>-8.96e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2007]</th>       <td> -1.24e+05</td> <td> 2.02e+04</td> <td>   -6.129</td> <td> 0.000</td> <td>-1.64e+05</td> <td>-8.44e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2008]</th>       <td>-1.051e+05</td> <td> 2.04e+04</td> <td>   -5.140</td> <td> 0.000</td> <td>-1.45e+05</td> <td> -6.5e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2009]</th>       <td>-9.142e+04</td> <td> 2.16e+04</td> <td>   -4.237</td> <td> 0.000</td> <td>-1.34e+05</td> <td>-4.91e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2010]</th>       <td>-8.297e+04</td> <td> 2.32e+04</td> <td>   -3.582</td> <td> 0.000</td> <td>-1.28e+05</td> <td>-3.76e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2011]</th>       <td>-9.734e+04</td> <td> 2.36e+04</td> <td>   -4.129</td> <td> 0.000</td> <td>-1.44e+05</td> <td>-5.11e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2012]</th>       <td>-9.427e+04</td> <td> 2.26e+04</td> <td>   -4.180</td> <td> 0.000</td> <td>-1.38e+05</td> <td>-5.01e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2013]</th>       <td>-7.083e+04</td> <td>  2.2e+04</td> <td>   -3.212</td> <td> 0.001</td> <td>-1.14e+05</td> <td>-2.76e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2014]</th>       <td>-8.733e+04</td> <td> 1.98e+04</td> <td>   -4.402</td> <td> 0.000</td> <td>-1.26e+05</td> <td>-4.84e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2015]</th>       <td>-1.156e+05</td> <td> 3.24e+04</td> <td>   -3.562</td> <td> 0.000</td> <td>-1.79e+05</td> <td> -5.2e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1934.0]</th> <td> 9.631e+04</td> <td> 1.65e+05</td> <td>    0.582</td> <td> 0.560</td> <td>-2.28e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1940.0]</th> <td>-9.117e+04</td> <td> 1.17e+05</td> <td>   -0.780</td> <td> 0.435</td> <td> -3.2e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1944.0]</th> <td> 3.074e+04</td> <td> 1.65e+05</td> <td>    0.186</td> <td> 0.852</td> <td>-2.93e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1945.0]</th> <td>-1.961e+04</td> <td> 9.58e+04</td> <td>   -0.205</td> <td> 0.838</td> <td>-2.07e+05</td> <td> 1.68e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1946.0]</th> <td> 5.975e+04</td> <td> 1.65e+05</td> <td>    0.363</td> <td> 0.717</td> <td>-2.63e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1948.0]</th> <td>-6.684e+04</td> <td> 1.66e+05</td> <td>   -0.404</td> <td> 0.686</td> <td>-3.91e+05</td> <td> 2.58e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1950.0]</th> <td>-3.581e+04</td> <td> 1.66e+05</td> <td>   -0.216</td> <td> 0.829</td> <td>-3.61e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1951.0]</th> <td> 5.426e+04</td> <td> 1.65e+05</td> <td>    0.328</td> <td> 0.743</td> <td> -2.7e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1953.0]</th> <td>-2.391e+05</td> <td> 1.67e+05</td> <td>   -1.435</td> <td> 0.151</td> <td>-5.66e+05</td> <td> 8.76e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1954.0]</th> <td> 2.908e+05</td> <td> 1.65e+05</td> <td>    1.763</td> <td> 0.078</td> <td>-3.25e+04</td> <td> 6.14e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1955.0]</th> <td>-2.857e+04</td> <td> 9.54e+04</td> <td>   -0.300</td> <td> 0.764</td> <td>-2.15e+05</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1956.0]</th> <td>-2.596e+05</td> <td> 9.61e+04</td> <td>   -2.701</td> <td> 0.007</td> <td>-4.48e+05</td> <td>-7.12e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1957.0]</th> <td>-8.782e+04</td> <td> 1.17e+05</td> <td>   -0.750</td> <td> 0.453</td> <td>-3.17e+05</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1958.0]</th> <td>-3.218e+04</td> <td> 9.64e+04</td> <td>   -0.334</td> <td> 0.738</td> <td>-2.21e+05</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1959.0]</th> <td>-1.517e+05</td> <td> 1.66e+05</td> <td>   -0.914</td> <td> 0.360</td> <td>-4.77e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1960.0]</th> <td>-5.086e+04</td> <td> 9.64e+04</td> <td>   -0.528</td> <td> 0.598</td> <td> -2.4e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1962.0]</th> <td>-3.445e+04</td> <td> 1.17e+05</td> <td>   -0.295</td> <td> 0.768</td> <td>-2.63e+05</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1963.0]</th> <td>-2.595e+05</td> <td>  8.3e+04</td> <td>   -3.127</td> <td> 0.002</td> <td>-4.22e+05</td> <td>-9.68e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1964.0]</th> <td>-4.204e+04</td> <td> 8.27e+04</td> <td>   -0.508</td> <td> 0.611</td> <td>-2.04e+05</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1965.0]</th> <td> 7.912e+04</td> <td> 8.29e+04</td> <td>    0.955</td> <td> 0.340</td> <td>-8.33e+04</td> <td> 2.42e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1967.0]</th> <td>-5.464e+04</td> <td> 1.17e+05</td> <td>   -0.468</td> <td> 0.640</td> <td>-2.83e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1968.0]</th> <td>-6.016e+04</td> <td> 6.75e+04</td> <td>   -0.891</td> <td> 0.373</td> <td>-1.93e+05</td> <td> 7.22e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1969.0]</th> <td>-5.924e+04</td> <td> 8.26e+04</td> <td>   -0.717</td> <td> 0.473</td> <td>-2.21e+05</td> <td> 1.03e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1970.0]</th> <td>-2.092e+05</td> <td> 5.51e+04</td> <td>   -3.796</td> <td> 0.000</td> <td>-3.17e+05</td> <td>-1.01e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1971.0]</th> <td> 1.026e+04</td> <td> 1.65e+05</td> <td>    0.062</td> <td> 0.951</td> <td>-3.14e+05</td> <td> 3.34e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1972.0]</th> <td>-1.633e+05</td> <td> 9.56e+04</td> <td>   -1.707</td> <td> 0.088</td> <td>-3.51e+05</td> <td> 2.42e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1973.0]</th> <td>-8.893e+04</td> <td> 8.27e+04</td> <td>   -1.076</td> <td> 0.282</td> <td>-2.51e+05</td> <td> 7.31e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1974.0]</th> <td> -954.8271</td> <td> 1.18e+05</td> <td>   -0.008</td> <td> 0.994</td> <td>-2.32e+05</td> <td>  2.3e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1975.0]</th> <td> -2.01e+04</td> <td> 7.37e+04</td> <td>   -0.273</td> <td> 0.785</td> <td>-1.65e+05</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1976.0]</th> <td>-2.115e+05</td> <td> 1.65e+05</td> <td>   -1.282</td> <td> 0.200</td> <td>-5.35e+05</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1977.0]</th> <td> 3.225e+04</td> <td> 6.24e+04</td> <td>    0.517</td> <td> 0.605</td> <td>   -9e+04</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1978.0]</th> <td>-1.822e+05</td> <td> 9.67e+04</td> <td>   -1.884</td> <td> 0.060</td> <td>-3.72e+05</td> <td> 7315.391</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1979.0]</th> <td> 3897.6815</td> <td> 6.33e+04</td> <td>    0.062</td> <td> 0.951</td> <td> -1.2e+05</td> <td> 1.28e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1980.0]</th> <td> 5.464e+04</td> <td> 6.27e+04</td> <td>    0.872</td> <td> 0.383</td> <td>-6.82e+04</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1981.0]</th> <td>-3.348e+04</td> <td> 8.24e+04</td> <td>   -0.406</td> <td> 0.685</td> <td>-1.95e+05</td> <td> 1.28e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1982.0]</th> <td> 3.491e+04</td> <td> 6.25e+04</td> <td>    0.559</td> <td> 0.576</td> <td>-8.76e+04</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1983.0]</th> <td>-5738.1791</td> <td> 4.27e+04</td> <td>   -0.134</td> <td> 0.893</td> <td>-8.95e+04</td> <td>  7.8e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1984.0]</th> <td>-7.665e+04</td> <td> 4.28e+04</td> <td>   -1.791</td> <td> 0.073</td> <td>-1.61e+05</td> <td> 7254.095</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1985.0]</th> <td>-1.029e+05</td> <td> 4.42e+04</td> <td>   -2.327</td> <td> 0.020</td> <td>-1.89e+05</td> <td>-1.62e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1986.0]</th> <td>-2.057e+05</td> <td> 4.43e+04</td> <td>   -4.646</td> <td> 0.000</td> <td>-2.93e+05</td> <td>-1.19e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1987.0]</th> <td> 4.494e+05</td> <td> 4.43e+04</td> <td>   10.156</td> <td> 0.000</td> <td> 3.63e+05</td> <td> 5.36e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1988.0]</th> <td> -4.68e+04</td> <td> 4.99e+04</td> <td>   -0.937</td> <td> 0.349</td> <td>-1.45e+05</td> <td> 5.11e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1989.0]</th> <td> 2.613e+04</td> <td> 3.81e+04</td> <td>    0.686</td> <td> 0.493</td> <td>-4.86e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1990.0]</th> <td> -8.74e+04</td> <td> 3.62e+04</td> <td>   -2.417</td> <td> 0.016</td> <td>-1.58e+05</td> <td>-1.65e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1991.0]</th> <td>-7.945e+04</td> <td> 4.14e+04</td> <td>   -1.917</td> <td> 0.055</td> <td>-1.61e+05</td> <td> 1787.447</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1992.0]</th> <td>-1.316e+04</td> <td> 4.77e+04</td> <td>   -0.276</td> <td> 0.783</td> <td>-1.07e+05</td> <td> 8.04e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1993.0]</th> <td> 6.466e+04</td> <td> 4.78e+04</td> <td>    1.353</td> <td> 0.176</td> <td> -2.9e+04</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1994.0]</th> <td>  1.59e+05</td> <td> 4.43e+04</td> <td>    3.588</td> <td> 0.000</td> <td> 7.21e+04</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1995.0]</th> <td> 1.377e+05</td> <td> 4.77e+04</td> <td>    2.886</td> <td> 0.004</td> <td> 4.42e+04</td> <td> 2.31e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1996.0]</th> <td>-8821.4853</td> <td> 5.23e+04</td> <td>   -0.169</td> <td> 0.866</td> <td>-1.11e+05</td> <td> 9.36e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1997.0]</th> <td>-9.629e+04</td> <td> 5.23e+04</td> <td>   -1.841</td> <td> 0.066</td> <td>-1.99e+05</td> <td> 6255.923</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1998.0]</th> <td>-1.897e+04</td> <td> 4.14e+04</td> <td>   -0.458</td> <td> 0.647</td> <td>   -1e+05</td> <td> 6.22e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1999.0]</th> <td> 1.264e+05</td> <td> 4.59e+04</td> <td>    2.756</td> <td> 0.006</td> <td> 3.65e+04</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2000.0]</th> <td>-5.066e+04</td> <td> 3.09e+04</td> <td>   -1.641</td> <td> 0.101</td> <td>-1.11e+05</td> <td> 9854.753</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2001.0]</th> <td> 2.028e+05</td> <td> 4.27e+04</td> <td>    4.747</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 2.87e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2002.0]</th> <td> 1.845e+05</td> <td> 4.02e+04</td> <td>    4.588</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2003.0]</th> <td> 7.542e+04</td> <td> 2.98e+04</td> <td>    2.531</td> <td> 0.011</td> <td>  1.7e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2004.0]</th> <td>-1781.1413</td> <td> 3.53e+04</td> <td>   -0.051</td> <td> 0.960</td> <td>-7.09e+04</td> <td> 6.73e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2005.0]</th> <td> 8.478e+04</td> <td> 3.09e+04</td> <td>    2.747</td> <td> 0.006</td> <td> 2.43e+04</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2006.0]</th> <td> 3.498e+04</td> <td>  3.7e+04</td> <td>    0.944</td> <td> 0.345</td> <td>-3.76e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2007.0]</th> <td> 4.727e+04</td> <td> 3.08e+04</td> <td>    1.536</td> <td> 0.124</td> <td> -1.3e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2008.0]</th> <td>  1.79e+05</td> <td> 4.42e+04</td> <td>    4.046</td> <td> 0.000</td> <td> 9.23e+04</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2009.0]</th> <td> 1.856e+05</td> <td> 3.61e+04</td> <td>    5.147</td> <td> 0.000</td> <td> 1.15e+05</td> <td> 2.56e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2010.0]</th> <td> 2.192e+05</td> <td> 4.27e+04</td> <td>    5.139</td> <td> 0.000</td> <td> 1.36e+05</td> <td> 3.03e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2011.0]</th> <td>-6.905e+04</td> <td> 5.51e+04</td> <td>   -1.253</td> <td> 0.210</td> <td>-1.77e+05</td> <td> 3.89e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2012.0]</th> <td> 5066.4086</td> <td> 5.84e+04</td> <td>    0.087</td> <td> 0.931</td> <td>-1.09e+05</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2013.0]</th> <td> 6.521e+04</td> <td> 2.98e+04</td> <td>    2.187</td> <td> 0.029</td> <td> 6760.894</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2014.0]</th> <td> 4.153e+04</td> <td> 1.95e+04</td> <td>    2.128</td> <td> 0.033</td> <td> 3269.731</td> <td> 7.98e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2015.0]</th> <td>-5.634e+04</td> <td> 4.42e+04</td> <td>   -1.276</td> <td> 0.202</td> <td>-1.43e+05</td> <td> 3.02e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>       <td> 3.646e+04</td> <td>  1.5e+04</td> <td>    2.434</td> <td> 0.015</td> <td> 7095.040</td> <td> 6.58e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>       <td>-2.043e+04</td> <td> 1.33e+04</td> <td>   -1.530</td> <td> 0.126</td> <td>-4.66e+04</td> <td> 5735.374</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>       <td> 7.214e+05</td> <td> 2.49e+04</td> <td>   28.927</td> <td> 0.000</td> <td> 6.73e+05</td> <td>  7.7e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>       <td> 2.565e+05</td> <td> 2.65e+04</td> <td>    9.670</td> <td> 0.000</td> <td> 2.05e+05</td> <td> 3.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>       <td> 2.496e+05</td> <td> 2.17e+04</td> <td>   11.483</td> <td> 0.000</td> <td> 2.07e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>       <td> 2.164e+05</td> <td> 2.74e+04</td> <td>    7.887</td> <td> 0.000</td> <td> 1.63e+05</td> <td>  2.7e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>       <td> 2.435e+05</td> <td> 2.61e+04</td> <td>    9.313</td> <td> 0.000</td> <td> 1.92e+05</td> <td> 2.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>       <td> 1.191e+05</td> <td> 2.38e+04</td> <td>    5.009</td> <td> 0.000</td> <td> 7.25e+04</td> <td> 1.66e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>       <td> 5.912e+04</td> <td>  3.4e+04</td> <td>    1.737</td> <td> 0.082</td> <td>-7611.314</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>       <td>  1.32e+05</td> <td>  3.9e+04</td> <td>    3.388</td> <td> 0.001</td> <td> 5.56e+04</td> <td> 2.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>       <td> 7.743e+04</td> <td> 3.72e+04</td> <td>    2.084</td> <td> 0.037</td> <td> 4591.037</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>       <td>  8.36e+04</td> <td>  2.1e+04</td> <td>    3.979</td> <td> 0.000</td> <td> 4.24e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>       <td>-5.118e+04</td> <td> 1.24e+04</td> <td>   -4.138</td> <td> 0.000</td> <td>-7.54e+04</td> <td>-2.69e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>       <td> 1.752e+05</td> <td>  3.5e+04</td> <td>    5.012</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 2.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>       <td>  1.57e+05</td> <td> 2.24e+04</td> <td>    7.014</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>       <td> 4.622e+04</td> <td> 3.31e+04</td> <td>    1.396</td> <td> 0.163</td> <td>-1.87e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>       <td>  2.13e+05</td> <td> 2.56e+04</td> <td>    8.322</td> <td> 0.000</td> <td> 1.63e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>       <td> 8181.6905</td> <td> 1.48e+04</td> <td>    0.553</td> <td> 0.580</td> <td>-2.08e+04</td> <td> 3.72e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>       <td> 1.342e+04</td> <td> 1.54e+04</td> <td>    0.868</td> <td> 0.385</td> <td>-1.69e+04</td> <td> 4.37e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>       <td>-1.179e+04</td> <td> 1.78e+04</td> <td>   -0.662</td> <td> 0.508</td> <td>-4.67e+04</td> <td> 2.31e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>       <td> 3.149e+05</td> <td> 2.84e+04</td> <td>   11.095</td> <td> 0.000</td> <td> 2.59e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>       <td>  1.51e+05</td> <td> 3.05e+04</td> <td>    4.950</td> <td> 0.000</td> <td> 9.12e+04</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>       <td> 6.756e+04</td> <td> 1.69e+04</td> <td>    4.003</td> <td> 0.000</td> <td> 3.45e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>       <td> 1.283e+06</td> <td> 3.35e+04</td> <td>   38.345</td> <td> 0.000</td> <td> 1.22e+06</td> <td> 1.35e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>       <td> 4.817e+05</td> <td> 2.19e+04</td> <td>   21.968</td> <td> 0.000</td> <td> 4.39e+05</td> <td> 5.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>       <td> 2.594e+04</td> <td> 1.43e+04</td> <td>    1.817</td> <td> 0.069</td> <td>-2040.310</td> <td> 5.39e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>       <td>  1.74e+05</td> <td> 3.17e+04</td> <td>    5.488</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>       <td> 1.864e+05</td> <td>  2.9e+04</td> <td>    6.421</td> <td> 0.000</td> <td> 1.29e+05</td> <td> 2.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>       <td> 1.777e+05</td> <td> 3.13e+04</td> <td>    5.685</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>       <td> 2.959e+04</td> <td> 1.73e+04</td> <td>    1.710</td> <td> 0.087</td> <td>-4321.028</td> <td> 6.35e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>       <td> 7.795e+04</td> <td> 1.89e+04</td> <td>    4.126</td> <td> 0.000</td> <td> 4.09e+04</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>       <td> 2.426e+04</td> <td> 1.64e+04</td> <td>    1.479</td> <td> 0.139</td> <td>-7899.363</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>       <td> 7.557e+04</td> <td> 1.85e+04</td> <td>    4.081</td> <td> 0.000</td> <td> 3.93e+04</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>       <td> 1.302e+05</td> <td>  2.9e+04</td> <td>    4.495</td> <td> 0.000</td> <td> 7.34e+04</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>       <td>-8.992e+04</td> <td> 2.39e+04</td> <td>   -3.763</td> <td> 0.000</td> <td>-1.37e+05</td> <td>-4.31e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>       <td> 9.551e+04</td> <td>  3.4e+04</td> <td>    2.809</td> <td> 0.005</td> <td> 2.89e+04</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>       <td> 1.516e+05</td> <td> 2.75e+04</td> <td>    5.518</td> <td> 0.000</td> <td> 9.77e+04</td> <td> 2.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>       <td> 1.701e+05</td> <td> 2.63e+04</td> <td>    6.468</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 2.22e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>       <td> 7.755e+04</td> <td> 3.55e+04</td> <td>    2.184</td> <td> 0.029</td> <td> 7948.816</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>       <td> -2.57e+04</td> <td> 1.36e+04</td> <td>   -1.893</td> <td> 0.058</td> <td>-5.23e+04</td> <td>  908.955</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>       <td> 3.967e+05</td> <td> 2.89e+04</td> <td>   13.706</td> <td> 0.000</td> <td>  3.4e+05</td> <td> 4.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>       <td> 2.278e+05</td> <td> 2.74e+04</td> <td>    8.301</td> <td> 0.000</td> <td> 1.74e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>       <td> 3.584e+05</td> <td> 2.81e+04</td> <td>   12.760</td> <td> 0.000</td> <td> 3.03e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>       <td> 6.847e+04</td> <td> 2.02e+04</td> <td>    3.391</td> <td> 0.001</td> <td> 2.89e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>       <td> 2.342e+05</td> <td> 2.82e+04</td> <td>    8.300</td> <td> 0.000</td> <td> 1.79e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>       <td> 4.622e+04</td> <td> 2.22e+04</td> <td>    2.083</td> <td> 0.037</td> <td> 2731.412</td> <td> 8.97e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>       <td> 3.931e+05</td> <td> 2.89e+04</td> <td>   13.624</td> <td> 0.000</td> <td> 3.37e+05</td> <td>  4.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>       <td> 5.105e+05</td> <td> 2.57e+04</td> <td>   19.835</td> <td> 0.000</td> <td>  4.6e+05</td> <td> 5.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>       <td> 2.252e+05</td> <td> 2.79e+04</td> <td>    8.062</td> <td> 0.000</td> <td>  1.7e+05</td> <td>  2.8e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>       <td> 2.057e+05</td> <td> 2.26e+04</td> <td>    9.112</td> <td> 0.000</td> <td> 1.61e+05</td> <td>  2.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>       <td> 1.979e+05</td> <td> 2.83e+04</td> <td>    6.995</td> <td> 0.000</td> <td> 1.42e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>       <td> 1.141e+05</td> <td> 1.97e+04</td> <td>    5.792</td> <td> 0.000</td> <td> 7.55e+04</td> <td> 1.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>       <td> 3.754e+05</td> <td> 2.73e+04</td> <td>   13.731</td> <td> 0.000</td> <td> 3.22e+05</td> <td> 4.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>       <td> 2.577e+05</td> <td> 2.45e+04</td> <td>   10.525</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 3.06e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>       <td> 1.102e+05</td> <td> 3.02e+04</td> <td>    3.648</td> <td> 0.000</td> <td>  5.1e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>       <td> 1.118e+05</td> <td> 2.07e+04</td> <td>    5.390</td> <td> 0.000</td> <td> 7.11e+04</td> <td> 1.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>       <td> 6.376e+04</td> <td> 3.12e+04</td> <td>    2.045</td> <td> 0.041</td> <td> 2636.234</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>       <td> 1.756e+05</td> <td> 2.12e+04</td> <td>    8.294</td> <td> 0.000</td> <td> 1.34e+05</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>       <td> 2.035e+05</td> <td> 2.27e+04</td> <td>    8.960</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 2.48e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>       <td> 5.865e+04</td> <td>  1.9e+04</td> <td>    3.087</td> <td> 0.002</td> <td> 2.14e+04</td> <td> 9.59e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>       <td> 3.345e+04</td> <td> 2.55e+04</td> <td>    1.314</td> <td> 0.189</td> <td>-1.65e+04</td> <td> 8.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>       <td>  5.96e+04</td> <td> 3.25e+04</td> <td>    1.835</td> <td> 0.067</td> <td>-4074.016</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>       <td> 1.601e+04</td> <td> 1.73e+04</td> <td>    0.925</td> <td> 0.355</td> <td>-1.79e+04</td> <td> 4.99e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>       <td> 1.661e+04</td> <td> 1.83e+04</td> <td>    0.909</td> <td> 0.363</td> <td>-1.92e+04</td> <td> 5.24e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>       <td> 1.362e+05</td> <td> 3.25e+04</td> <td>    4.189</td> <td> 0.000</td> <td> 7.25e+04</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>       <td> 2277.0673</td> <td> 1.89e+04</td> <td>    0.121</td> <td> 0.904</td> <td>-3.47e+04</td> <td> 3.93e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>       <td> 8578.8637</td> <td> 1.91e+04</td> <td>    0.448</td> <td> 0.654</td> <td> -2.9e+04</td> <td> 4.61e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>       <td>-1143.8310</td> <td> 1.45e+04</td> <td>   -0.079</td> <td> 0.937</td> <td>-2.95e+04</td> <td> 2.72e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>       <td> 2.662e+05</td> <td> 2.68e+04</td> <td>    9.937</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 3.19e+05</td>
</tr>
<tr>
  <th>bedrooms</th>                  <td>-3.232e+04</td> <td> 1711.610</td> <td>  -18.882</td> <td> 0.000</td> <td>-3.57e+04</td> <td> -2.9e+04</td>
</tr>
<tr>
  <th>bathrooms</th>                 <td> 2.168e+04</td> <td> 2777.151</td> <td>    7.807</td> <td> 0.000</td> <td> 1.62e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>sqft_living</th>               <td>  194.6130</td> <td>    2.711</td> <td>   71.783</td> <td> 0.000</td> <td>  189.299</td> <td>  199.927</td>
</tr>
<tr>
  <th>sqft_lot</th>                  <td>    0.1898</td> <td>    0.051</td> <td>    3.688</td> <td> 0.000</td> <td>    0.089</td> <td>    0.291</td>
</tr>
<tr>
  <th>floors</th>                    <td>-2.405e+04</td> <td> 3229.987</td> <td>   -7.444</td> <td> 0.000</td> <td>-3.04e+04</td> <td>-1.77e+04</td>
</tr>
<tr>
  <th>grade</th>                     <td> 7.527e+04</td> <td> 1829.495</td> <td>   41.145</td> <td> 0.000</td> <td> 7.17e+04</td> <td> 7.89e+04</td>
</tr>
<tr>
  <th>lat</th>                       <td> 1.555e+05</td> <td>  6.8e+04</td> <td>    2.288</td> <td> 0.022</td> <td> 2.23e+04</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>long</th>                      <td>-1.922e+05</td> <td> 4.93e+04</td> <td>   -3.901</td> <td> 0.000</td> <td>-2.89e+05</td> <td>-9.56e+04</td>
</tr>
<tr>
  <th>sqft_lot15</th>                <td>    0.4128</td> <td>    0.161</td> <td>    2.572</td> <td> 0.010</td> <td>    0.098</td> <td>    0.727</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>18978.217</td> <th>  Durbin-Watson:     </th>  <td>   1.989</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>2734243.685</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.810</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>58.299</td>   <th>  Cond. No.          </th>  <td>1.82e+08</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.82e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# In this model there is an R2 value of 0.804. Much better fit. 

# The P value for 'condition' variable 2 is 0.106
# The P value for 'condition' variable 3 is 0.131
# This could be because there were so few houses in that codition to draw info from. 

# The P values for 'yr_built' from 1902-1928, 1930-1932, 1934-1945, 1949-1950 all are well over 0.05. 
# The P values for 'yr_renovated' from 1934-1955, 1957-1962, 1964-1969, 1971-1984, 1988-1989, 1991-1993, 
# 1996-1998, 2000, 2004, 2006-2007, 2011-2012, and 2015 are all well over 0.05.
# This could be because there were little to no houses renovated during those years. 

# The P values in 'zipcode' 98003, 98011, 98028, 98030-98032, 98042, 98055, 98058, 98092, 
# 98148, 98155, 98166, 98168, 98178, 98188, 98198 all are well over 0.05. 
# This could be because of lack of data from housing in those zipcodes. 

# Notice the P value for sqft_lot that was problematic before is now fine. 

# Our skew is up to 3.8, from 3.4 in the last model. Interesting. 
# Kurtosis is up to 58.3 from 43.2 in the last model.
# The condition number is down to 1.82^8 from 2.06^8 in the last model. 
```

From the results of both models, there was a clear improvement overall in ability to predict housing prices. My next thoughts would be to drop some of the columns causing multicollinearity in addition to scaling some of the data units to match up better. 

## Data transformations 2 - feature scaling.

### Normalize with min max feature scaling to manage the difference in magnitude. 

We are going to look at features that have units of measure that are very different from most of the other variables and look at scaling them to match better for a better fit model. One of the ways we can do this is by using min max scaling to adjust the magnitude of the units down to range 0-1.


```python
# Take a look at the histogram to show the difference in magnitude of our numerical features. 
df_filtered1.hist(figsize=(12,12));
```


![png](output_145_0.png)



```python
# The features we'll want to scale will be:
# sqft living, sqft lot, lat, long, sqft lot 15
```


```python
# this loops through the cols needing to be scaled 
# applies equation for min max scaling and adds new col to dataframe

scale = ['sqft_living', 'sqft_lot', 'lat', 'long', 'sqft_lot15']
for col in scale:
    x = df_filtered1[col]
    scaled_x = (x - min(x)) / (max(x) - min(x))
    df_filtered1['scaled_' + col] = scaled_x

df_filtered1.head()
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
      <td>0.069349</td>
      <td>0.004406</td>
      <td>0.571498</td>
      <td>0.214345</td>
      <td>0.053174</td>
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
      <td>0.188356</td>
      <td>0.005774</td>
      <td>0.908959</td>
      <td>0.162636</td>
      <td>0.074331</td>
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
      <td>0.034247</td>
      <td>0.008142</td>
      <td>0.936143</td>
      <td>0.234362</td>
      <td>0.078830</td>
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
      <td>0.136130</td>
      <td>0.003848</td>
      <td>0.586939</td>
      <td>0.100917</td>
      <td>0.046260</td>
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
      <td>0.112158</td>
      <td>0.006493</td>
      <td>0.741354</td>
      <td>0.391159</td>
      <td>0.072884</td>
    </tr>
  </tbody>
</table>
</div>



The scale columns are added to the existing dataframe and can then be applied into the model the same as the previous columns, just in place of the originals to show the difference in model performance. 

### Check hist to show new scaled data. 


```python
df_filtered1.hist(figsize=(12,12));
```


![png](output_150_0.png)


# MODEL

## Create model. 

Next we create an updated model that has the scaled numerical features with similar magnitudes to rest of the data in place of previous ones. The features we scaled were sqft_living, sqft_lot, lat, long, sqft_lot15.


```python
df_filtered1.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated',
           'zipcode', 'lat', 'long', 'sqft_lot15', 'scaled_sqft_living',
           'scaled_sqft_lot', 'scaled_lat', 'scaled_long', 'scaled_sqft_lot15'],
          dtype='object')




```python
# create model using list of predictors by typing them all out. 
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'scaled_sqft_living', 'scaled_sqft_lot', 'floors',
       'C(waterfront)', 'C(condition)', 'grade', 'C(yr_built)', 'C(yr_renovated)',
       'C(zipcode)', 'scaled_lat', 'scaled_long', 'scaled_sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = smf.ols(formula=formula, data=df_filtered1).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.804</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.801</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   318.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 19 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>18:53:04</td>     <th>  Log-Likelihood:    </th> <td>-2.8262e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21059</td>      <th>  AIC:               </th>  <td>5.658e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20791</td>      <th>  BIC:               </th>  <td>5.679e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>   267</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>-4.098e+05</td> <td> 4.14e+04</td> <td>   -9.892</td> <td> 0.000</td> <td>-4.91e+05</td> <td>-3.29e+05</td>
</tr>
<tr>
  <th>C(waterfront)[T.1.0]</th>      <td> 8.542e+05</td> <td> 1.45e+04</td> <td>   59.037</td> <td> 0.000</td> <td> 8.26e+05</td> <td> 8.83e+05</td>
</tr>
<tr>
  <th>C(condition)[T.2]</th>         <td> 5.493e+04</td> <td>  3.4e+04</td> <td>    1.616</td> <td> 0.106</td> <td>-1.17e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>C(condition)[T.3]</th>         <td> 4.746e+04</td> <td> 3.14e+04</td> <td>    1.510</td> <td> 0.131</td> <td>-1.41e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>C(condition)[T.4]</th>         <td> 6.917e+04</td> <td> 3.14e+04</td> <td>    2.200</td> <td> 0.028</td> <td> 7547.236</td> <td> 1.31e+05</td>
</tr>
<tr>
  <th>C(condition)[T.5]</th>         <td> 1.031e+05</td> <td> 3.16e+04</td> <td>    3.262</td> <td> 0.001</td> <td> 4.12e+04</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1901]</th>       <td>-1.117e+05</td> <td> 3.54e+04</td> <td>   -3.153</td> <td> 0.002</td> <td>-1.81e+05</td> <td>-4.23e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1902]</th>       <td>-6.575e+04</td> <td> 3.65e+04</td> <td>   -1.800</td> <td> 0.072</td> <td>-1.37e+05</td> <td> 5861.110</td>
</tr>
<tr>
  <th>C(yr_built)[T.1903]</th>       <td>-4.726e+04</td> <td> 3.04e+04</td> <td>   -1.555</td> <td> 0.120</td> <td>-1.07e+05</td> <td> 1.23e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1904]</th>       <td>-1.861e+04</td> <td> 3.07e+04</td> <td>   -0.607</td> <td> 0.544</td> <td>-7.87e+04</td> <td> 4.15e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1905]</th>       <td> 1.254e+04</td> <td> 2.64e+04</td> <td>    0.474</td> <td> 0.635</td> <td>-3.93e+04</td> <td> 6.43e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1906]</th>       <td>-2.187e+04</td> <td> 2.49e+04</td> <td>   -0.878</td> <td> 0.380</td> <td>-7.07e+04</td> <td> 2.69e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1907]</th>       <td>-7453.1960</td> <td> 2.72e+04</td> <td>   -0.274</td> <td> 0.784</td> <td>-6.09e+04</td> <td> 4.59e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1908]</th>       <td>-4.597e+04</td> <td> 2.53e+04</td> <td>   -1.815</td> <td> 0.069</td> <td>-9.56e+04</td> <td> 3667.808</td>
</tr>
<tr>
  <th>C(yr_built)[T.1909]</th>       <td>-1813.8779</td> <td> 2.48e+04</td> <td>   -0.073</td> <td> 0.942</td> <td>-5.05e+04</td> <td> 4.68e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1910]</th>       <td> 1.118e+04</td> <td> 2.31e+04</td> <td>    0.484</td> <td> 0.628</td> <td>-3.41e+04</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1911]</th>       <td>-3555.5462</td> <td> 2.66e+04</td> <td>   -0.134</td> <td> 0.894</td> <td>-5.58e+04</td> <td> 4.86e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1912]</th>       <td>-1.461e+04</td> <td> 2.62e+04</td> <td>   -0.558</td> <td> 0.577</td> <td>-6.59e+04</td> <td> 3.67e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1913]</th>       <td>-8219.8580</td> <td> 2.86e+04</td> <td>   -0.287</td> <td> 0.774</td> <td>-6.44e+04</td> <td> 4.79e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1914]</th>       <td>-1.834e+04</td> <td> 2.89e+04</td> <td>   -0.634</td> <td> 0.526</td> <td>-7.51e+04</td> <td> 3.84e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1915]</th>       <td>-1.866e+04</td> <td> 2.76e+04</td> <td>   -0.675</td> <td> 0.500</td> <td>-7.28e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1916]</th>       <td> 6124.1018</td> <td> 2.59e+04</td> <td>    0.237</td> <td> 0.813</td> <td>-4.45e+04</td> <td> 5.68e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1917]</th>       <td>-2.268e+04</td> <td> 2.84e+04</td> <td>   -0.798</td> <td> 0.425</td> <td>-7.84e+04</td> <td>  3.3e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1918]</th>       <td>-2.217e+04</td> <td> 2.36e+04</td> <td>   -0.940</td> <td> 0.347</td> <td>-6.84e+04</td> <td> 2.41e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1919]</th>       <td>-2.643e+04</td> <td> 2.54e+04</td> <td>   -1.042</td> <td> 0.297</td> <td>-7.62e+04</td> <td> 2.33e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1920]</th>       <td> -2.18e+04</td> <td> 2.46e+04</td> <td>   -0.885</td> <td> 0.376</td> <td>-7.01e+04</td> <td> 2.65e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1921]</th>       <td>-1.642e+04</td> <td> 2.65e+04</td> <td>   -0.620</td> <td> 0.535</td> <td>-6.83e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1922]</th>       <td>-5156.4625</td> <td>  2.5e+04</td> <td>   -0.207</td> <td> 0.836</td> <td>-5.41e+04</td> <td> 4.38e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1923]</th>       <td>-9454.6436</td> <td> 2.57e+04</td> <td>   -0.369</td> <td> 0.712</td> <td>-5.97e+04</td> <td> 4.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1924]</th>       <td>-2.055e+04</td> <td> 2.29e+04</td> <td>   -0.898</td> <td> 0.369</td> <td>-6.54e+04</td> <td> 2.43e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1925]</th>       <td>-3.809e+04</td> <td> 2.22e+04</td> <td>   -1.714</td> <td> 0.087</td> <td>-8.17e+04</td> <td> 5479.639</td>
</tr>
<tr>
  <th>C(yr_built)[T.1926]</th>       <td>-8127.6901</td> <td> 2.18e+04</td> <td>   -0.372</td> <td> 0.710</td> <td>-5.09e+04</td> <td> 3.47e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1927]</th>       <td>-2.573e+04</td> <td> 2.38e+04</td> <td>   -1.080</td> <td> 0.280</td> <td>-7.24e+04</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1928]</th>       <td> 5767.8989</td> <td> 2.33e+04</td> <td>    0.247</td> <td> 0.805</td> <td>   -4e+04</td> <td> 5.15e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1929]</th>       <td>-5.626e+04</td> <td> 2.38e+04</td> <td>   -2.363</td> <td> 0.018</td> <td>-1.03e+05</td> <td>-9593.008</td>
</tr>
<tr>
  <th>C(yr_built)[T.1930]</th>       <td>-2.468e+04</td> <td>  2.5e+04</td> <td>   -0.985</td> <td> 0.324</td> <td>-7.38e+04</td> <td> 2.44e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1931]</th>       <td> -3.38e+04</td> <td> 2.79e+04</td> <td>   -1.211</td> <td> 0.226</td> <td>-8.85e+04</td> <td> 2.09e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1932]</th>       <td>-4259.9025</td> <td> 3.38e+04</td> <td>   -0.126</td> <td> 0.900</td> <td>-7.04e+04</td> <td> 6.19e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1933]</th>       <td> 9.721e+04</td> <td> 3.67e+04</td> <td>    2.645</td> <td> 0.008</td> <td> 2.52e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1934]</th>       <td> 9180.6924</td> <td> 4.24e+04</td> <td>    0.216</td> <td> 0.829</td> <td> -7.4e+04</td> <td> 9.24e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1935]</th>       <td> 1.262e+04</td> <td> 3.94e+04</td> <td>    0.320</td> <td> 0.749</td> <td>-6.46e+04</td> <td> 8.98e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1936]</th>       <td> 5.333e+04</td> <td>  3.3e+04</td> <td>    1.618</td> <td> 0.106</td> <td>-1.13e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1937]</th>       <td>-8438.2796</td> <td> 2.74e+04</td> <td>   -0.308</td> <td> 0.758</td> <td>-6.21e+04</td> <td> 4.52e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1938]</th>       <td>-1928.1473</td> <td> 2.95e+04</td> <td>   -0.065</td> <td> 0.948</td> <td>-5.98e+04</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1939]</th>       <td> 4206.6894</td> <td> 2.44e+04</td> <td>    0.172</td> <td> 0.863</td> <td>-4.36e+04</td> <td> 5.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1940]</th>       <td>-1.219e+04</td> <td> 2.26e+04</td> <td>   -0.540</td> <td> 0.589</td> <td>-5.65e+04</td> <td> 3.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1941]</th>       <td> 7778.1663</td> <td> 2.23e+04</td> <td>    0.348</td> <td> 0.728</td> <td> -3.6e+04</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1942]</th>       <td> -1.48e+04</td> <td> 2.14e+04</td> <td>   -0.693</td> <td> 0.488</td> <td>-5.67e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1943]</th>       <td>-1.697e+04</td> <td> 2.23e+04</td> <td>   -0.762</td> <td> 0.446</td> <td>-6.06e+04</td> <td> 2.67e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1944]</th>       <td>-1.312e+04</td> <td> 2.31e+04</td> <td>   -0.567</td> <td> 0.571</td> <td>-5.84e+04</td> <td> 3.22e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1945]</th>       <td>-4677.9948</td> <td> 2.49e+04</td> <td>   -0.188</td> <td> 0.851</td> <td>-5.36e+04</td> <td> 4.42e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1946]</th>       <td>-4.724e+04</td> <td> 2.35e+04</td> <td>   -2.014</td> <td> 0.044</td> <td>-9.32e+04</td> <td>-1254.683</td>
</tr>
<tr>
  <th>C(yr_built)[T.1947]</th>       <td>-4.661e+04</td> <td> 2.09e+04</td> <td>   -2.231</td> <td> 0.026</td> <td>-8.76e+04</td> <td>-5660.071</td>
</tr>
<tr>
  <th>C(yr_built)[T.1948]</th>       <td>-4.518e+04</td> <td> 2.12e+04</td> <td>   -2.131</td> <td> 0.033</td> <td>-8.67e+04</td> <td>-3632.491</td>
</tr>
<tr>
  <th>C(yr_built)[T.1949]</th>       <td>-1.812e+04</td> <td> 2.18e+04</td> <td>   -0.832</td> <td> 0.406</td> <td>-6.08e+04</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1950]</th>       <td>-3.551e+04</td> <td> 2.11e+04</td> <td>   -1.687</td> <td> 0.092</td> <td>-7.68e+04</td> <td> 5756.453</td>
</tr>
<tr>
  <th>C(yr_built)[T.1951]</th>       <td>-5.029e+04</td> <td> 2.12e+04</td> <td>   -2.372</td> <td> 0.018</td> <td>-9.18e+04</td> <td>-8730.466</td>
</tr>
<tr>
  <th>C(yr_built)[T.1952]</th>       <td>-7.093e+04</td> <td> 2.14e+04</td> <td>   -3.318</td> <td> 0.001</td> <td>-1.13e+05</td> <td> -2.9e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1953]</th>       <td>-8.853e+04</td> <td> 2.14e+04</td> <td>   -4.128</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-4.65e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1954]</th>       <td> -7.25e+04</td> <td> 2.06e+04</td> <td>   -3.524</td> <td> 0.000</td> <td>-1.13e+05</td> <td>-3.22e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1955]</th>       <td>-9.017e+04</td> <td> 2.09e+04</td> <td>   -4.318</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-4.92e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1956]</th>       <td>-7.487e+04</td> <td> 2.18e+04</td> <td>   -3.431</td> <td> 0.001</td> <td>-1.18e+05</td> <td>-3.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1957]</th>       <td>-7.779e+04</td> <td> 2.17e+04</td> <td>   -3.583</td> <td> 0.000</td> <td> -1.2e+05</td> <td>-3.52e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1958]</th>       <td>-7.085e+04</td> <td> 2.14e+04</td> <td>   -3.314</td> <td> 0.001</td> <td>-1.13e+05</td> <td>-2.89e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1959]</th>       <td>-7.913e+04</td> <td> 2.04e+04</td> <td>   -3.877</td> <td> 0.000</td> <td>-1.19e+05</td> <td>-3.91e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1960]</th>       <td>-9.054e+04</td> <td> 2.11e+04</td> <td>   -4.284</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-4.91e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1961]</th>       <td>-8.844e+04</td> <td> 2.13e+04</td> <td>   -4.145</td> <td> 0.000</td> <td> -1.3e+05</td> <td>-4.66e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1962]</th>       <td>-9.185e+04</td> <td> 2.06e+04</td> <td>   -4.464</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-5.15e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1963]</th>       <td>-9.226e+04</td> <td> 2.11e+04</td> <td>   -4.383</td> <td> 0.000</td> <td>-1.34e+05</td> <td> -5.1e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1964]</th>       <td>-1.179e+05</td> <td> 2.23e+04</td> <td>   -5.294</td> <td> 0.000</td> <td>-1.62e+05</td> <td>-7.43e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1965]</th>       <td>-1.212e+05</td> <td> 2.19e+04</td> <td>   -5.534</td> <td> 0.000</td> <td>-1.64e+05</td> <td>-7.83e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1966]</th>       <td>-1.184e+05</td> <td> 2.12e+04</td> <td>   -5.590</td> <td> 0.000</td> <td> -1.6e+05</td> <td>-7.69e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1967]</th>       <td>-9.218e+04</td> <td> 2.04e+04</td> <td>   -4.521</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-5.22e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1968]</th>       <td>-8.354e+04</td> <td> 2.02e+04</td> <td>   -4.138</td> <td> 0.000</td> <td>-1.23e+05</td> <td> -4.4e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1969]</th>       <td>-5.168e+04</td> <td> 2.09e+04</td> <td>   -2.478</td> <td> 0.013</td> <td>-9.26e+04</td> <td>-1.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1970]</th>       <td>-8.579e+04</td> <td> 2.33e+04</td> <td>   -3.677</td> <td> 0.000</td> <td>-1.32e+05</td> <td>-4.01e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1971]</th>       <td>-7.506e+04</td> <td> 2.46e+04</td> <td>   -3.046</td> <td> 0.002</td> <td>-1.23e+05</td> <td>-2.68e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1972]</th>       <td>-1.078e+05</td> <td>  2.3e+04</td> <td>   -4.697</td> <td> 0.000</td> <td>-1.53e+05</td> <td>-6.28e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1973]</th>       <td>-9.692e+04</td> <td> 2.29e+04</td> <td>   -4.236</td> <td> 0.000</td> <td>-1.42e+05</td> <td>-5.21e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1974]</th>       <td>-1.184e+05</td> <td> 2.25e+04</td> <td>   -5.265</td> <td> 0.000</td> <td>-1.63e+05</td> <td>-7.44e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1975]</th>       <td>-1.031e+05</td> <td>  2.2e+04</td> <td>   -4.690</td> <td> 0.000</td> <td>-1.46e+05</td> <td>   -6e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1976]</th>       <td>-1.134e+05</td> <td> 2.11e+04</td> <td>   -5.369</td> <td> 0.000</td> <td>-1.55e+05</td> <td> -7.2e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1977]</th>       <td>-1.202e+05</td> <td> 2.01e+04</td> <td>   -5.988</td> <td> 0.000</td> <td> -1.6e+05</td> <td>-8.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1978]</th>       <td>-1.301e+05</td> <td> 2.02e+04</td> <td>   -6.435</td> <td> 0.000</td> <td> -1.7e+05</td> <td>-9.05e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1979]</th>       <td>-1.404e+05</td> <td> 2.05e+04</td> <td>   -6.864</td> <td> 0.000</td> <td> -1.8e+05</td> <td>   -1e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1980]</th>       <td>-9.825e+04</td> <td> 2.14e+04</td> <td>   -4.600</td> <td> 0.000</td> <td> -1.4e+05</td> <td>-5.64e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1981]</th>       <td>-7.246e+04</td> <td> 2.19e+04</td> <td>   -3.308</td> <td> 0.001</td> <td>-1.15e+05</td> <td>-2.95e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1982]</th>       <td>-7.788e+04</td> <td> 2.48e+04</td> <td>   -3.146</td> <td> 0.002</td> <td>-1.26e+05</td> <td>-2.94e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1983]</th>       <td>-7.506e+04</td> <td> 2.17e+04</td> <td>   -3.462</td> <td> 0.001</td> <td>-1.18e+05</td> <td>-3.26e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1984]</th>       <td>-1.078e+05</td> <td> 2.15e+04</td> <td>   -5.020</td> <td> 0.000</td> <td> -1.5e+05</td> <td>-6.57e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1985]</th>       <td>-9.765e+04</td> <td> 2.15e+04</td> <td>   -4.537</td> <td> 0.000</td> <td> -1.4e+05</td> <td>-5.55e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1986]</th>       <td>-1.122e+05</td> <td> 2.16e+04</td> <td>   -5.185</td> <td> 0.000</td> <td>-1.55e+05</td> <td>-6.98e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1987]</th>       <td>-1.249e+05</td> <td> 2.09e+04</td> <td>   -5.977</td> <td> 0.000</td> <td>-1.66e+05</td> <td>-8.39e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1988]</th>       <td>-1.195e+05</td> <td> 2.11e+04</td> <td>   -5.651</td> <td> 0.000</td> <td>-1.61e+05</td> <td> -7.8e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1989]</th>       <td>-1.075e+05</td> <td>  2.1e+04</td> <td>   -5.108</td> <td> 0.000</td> <td>-1.49e+05</td> <td>-6.62e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1990]</th>       <td>-1.453e+05</td> <td> 2.08e+04</td> <td>   -6.971</td> <td> 0.000</td> <td>-1.86e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>C(yr_built)[T.1991]</th>       <td>-1.035e+05</td> <td> 2.18e+04</td> <td>   -4.751</td> <td> 0.000</td> <td>-1.46e+05</td> <td>-6.08e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1992]</th>       <td>-1.213e+05</td> <td> 2.21e+04</td> <td>   -5.495</td> <td> 0.000</td> <td>-1.65e+05</td> <td> -7.8e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1993]</th>       <td>-1.088e+05</td> <td>  2.2e+04</td> <td>   -4.938</td> <td> 0.000</td> <td>-1.52e+05</td> <td>-6.56e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1994]</th>       <td>-1.081e+05</td> <td> 2.14e+04</td> <td>   -5.056</td> <td> 0.000</td> <td> -1.5e+05</td> <td>-6.62e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1995]</th>       <td> -1.03e+05</td> <td> 2.26e+04</td> <td>   -4.558</td> <td> 0.000</td> <td>-1.47e+05</td> <td>-5.87e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1996]</th>       <td>-1.007e+05</td> <td> 2.21e+04</td> <td>   -4.548</td> <td> 0.000</td> <td>-1.44e+05</td> <td>-5.73e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1997]</th>       <td>-1.188e+05</td> <td> 2.25e+04</td> <td>   -5.274</td> <td> 0.000</td> <td>-1.63e+05</td> <td>-7.46e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1998]</th>       <td>-1.227e+05</td> <td> 2.16e+04</td> <td>   -5.693</td> <td> 0.000</td> <td>-1.65e+05</td> <td>-8.05e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.1999]</th>       <td>  -8.7e+04</td> <td> 2.13e+04</td> <td>   -4.082</td> <td> 0.000</td> <td>-1.29e+05</td> <td>-4.52e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2000]</th>       <td>-1.042e+05</td> <td> 2.18e+04</td> <td>   -4.773</td> <td> 0.000</td> <td>-1.47e+05</td> <td>-6.14e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2001]</th>       <td>-9.635e+04</td> <td>  2.1e+04</td> <td>   -4.598</td> <td> 0.000</td> <td>-1.37e+05</td> <td>-5.53e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2002]</th>       <td>-1.196e+05</td> <td> 2.17e+04</td> <td>   -5.504</td> <td> 0.000</td> <td>-1.62e+05</td> <td> -7.7e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2003]</th>       <td>-1.051e+05</td> <td> 2.03e+04</td> <td>   -5.179</td> <td> 0.000</td> <td>-1.45e+05</td> <td>-6.53e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2004]</th>       <td>-1.134e+05</td> <td> 2.03e+04</td> <td>   -5.586</td> <td> 0.000</td> <td>-1.53e+05</td> <td>-7.36e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2005]</th>       <td> -1.27e+05</td> <td> 2.02e+04</td> <td>   -6.294</td> <td> 0.000</td> <td>-1.66e+05</td> <td>-8.74e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2006]</th>       <td>-1.291e+05</td> <td> 2.01e+04</td> <td>   -6.409</td> <td> 0.000</td> <td>-1.69e+05</td> <td>-8.96e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2007]</th>       <td> -1.24e+05</td> <td> 2.02e+04</td> <td>   -6.129</td> <td> 0.000</td> <td>-1.64e+05</td> <td>-8.44e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2008]</th>       <td>-1.051e+05</td> <td> 2.04e+04</td> <td>   -5.140</td> <td> 0.000</td> <td>-1.45e+05</td> <td> -6.5e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2009]</th>       <td>-9.142e+04</td> <td> 2.16e+04</td> <td>   -4.237</td> <td> 0.000</td> <td>-1.34e+05</td> <td>-4.91e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2010]</th>       <td>-8.297e+04</td> <td> 2.32e+04</td> <td>   -3.582</td> <td> 0.000</td> <td>-1.28e+05</td> <td>-3.76e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2011]</th>       <td>-9.734e+04</td> <td> 2.36e+04</td> <td>   -4.129</td> <td> 0.000</td> <td>-1.44e+05</td> <td>-5.11e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2012]</th>       <td>-9.427e+04</td> <td> 2.26e+04</td> <td>   -4.180</td> <td> 0.000</td> <td>-1.38e+05</td> <td>-5.01e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2013]</th>       <td>-7.083e+04</td> <td>  2.2e+04</td> <td>   -3.212</td> <td> 0.001</td> <td>-1.14e+05</td> <td>-2.76e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2014]</th>       <td>-8.733e+04</td> <td> 1.98e+04</td> <td>   -4.402</td> <td> 0.000</td> <td>-1.26e+05</td> <td>-4.84e+04</td>
</tr>
<tr>
  <th>C(yr_built)[T.2015]</th>       <td>-1.156e+05</td> <td> 3.24e+04</td> <td>   -3.562</td> <td> 0.000</td> <td>-1.79e+05</td> <td> -5.2e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1934.0]</th> <td> 9.631e+04</td> <td> 1.65e+05</td> <td>    0.582</td> <td> 0.560</td> <td>-2.28e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1940.0]</th> <td>-9.117e+04</td> <td> 1.17e+05</td> <td>   -0.780</td> <td> 0.435</td> <td> -3.2e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1944.0]</th> <td> 3.074e+04</td> <td> 1.65e+05</td> <td>    0.186</td> <td> 0.852</td> <td>-2.93e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1945.0]</th> <td>-1.961e+04</td> <td> 9.58e+04</td> <td>   -0.205</td> <td> 0.838</td> <td>-2.07e+05</td> <td> 1.68e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1946.0]</th> <td> 5.975e+04</td> <td> 1.65e+05</td> <td>    0.363</td> <td> 0.717</td> <td>-2.63e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1948.0]</th> <td>-6.684e+04</td> <td> 1.66e+05</td> <td>   -0.404</td> <td> 0.686</td> <td>-3.91e+05</td> <td> 2.58e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1950.0]</th> <td>-3.581e+04</td> <td> 1.66e+05</td> <td>   -0.216</td> <td> 0.829</td> <td>-3.61e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1951.0]</th> <td> 5.426e+04</td> <td> 1.65e+05</td> <td>    0.328</td> <td> 0.743</td> <td> -2.7e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1953.0]</th> <td>-2.391e+05</td> <td> 1.67e+05</td> <td>   -1.435</td> <td> 0.151</td> <td>-5.66e+05</td> <td> 8.76e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1954.0]</th> <td> 2.908e+05</td> <td> 1.65e+05</td> <td>    1.763</td> <td> 0.078</td> <td>-3.25e+04</td> <td> 6.14e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1955.0]</th> <td>-2.857e+04</td> <td> 9.54e+04</td> <td>   -0.300</td> <td> 0.764</td> <td>-2.15e+05</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1956.0]</th> <td>-2.596e+05</td> <td> 9.61e+04</td> <td>   -2.701</td> <td> 0.007</td> <td>-4.48e+05</td> <td>-7.12e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1957.0]</th> <td>-8.782e+04</td> <td> 1.17e+05</td> <td>   -0.750</td> <td> 0.453</td> <td>-3.17e+05</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1958.0]</th> <td>-3.218e+04</td> <td> 9.64e+04</td> <td>   -0.334</td> <td> 0.738</td> <td>-2.21e+05</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1959.0]</th> <td>-1.517e+05</td> <td> 1.66e+05</td> <td>   -0.914</td> <td> 0.360</td> <td>-4.77e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1960.0]</th> <td>-5.086e+04</td> <td> 9.64e+04</td> <td>   -0.528</td> <td> 0.598</td> <td> -2.4e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1962.0]</th> <td>-3.445e+04</td> <td> 1.17e+05</td> <td>   -0.295</td> <td> 0.768</td> <td>-2.63e+05</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1963.0]</th> <td>-2.595e+05</td> <td>  8.3e+04</td> <td>   -3.127</td> <td> 0.002</td> <td>-4.22e+05</td> <td>-9.68e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1964.0]</th> <td>-4.204e+04</td> <td> 8.27e+04</td> <td>   -0.508</td> <td> 0.611</td> <td>-2.04e+05</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1965.0]</th> <td> 7.912e+04</td> <td> 8.29e+04</td> <td>    0.955</td> <td> 0.340</td> <td>-8.33e+04</td> <td> 2.42e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1967.0]</th> <td>-5.464e+04</td> <td> 1.17e+05</td> <td>   -0.468</td> <td> 0.640</td> <td>-2.83e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1968.0]</th> <td>-6.016e+04</td> <td> 6.75e+04</td> <td>   -0.891</td> <td> 0.373</td> <td>-1.93e+05</td> <td> 7.22e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1969.0]</th> <td>-5.924e+04</td> <td> 8.26e+04</td> <td>   -0.717</td> <td> 0.473</td> <td>-2.21e+05</td> <td> 1.03e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1970.0]</th> <td>-2.092e+05</td> <td> 5.51e+04</td> <td>   -3.796</td> <td> 0.000</td> <td>-3.17e+05</td> <td>-1.01e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1971.0]</th> <td> 1.026e+04</td> <td> 1.65e+05</td> <td>    0.062</td> <td> 0.951</td> <td>-3.14e+05</td> <td> 3.34e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1972.0]</th> <td>-1.633e+05</td> <td> 9.56e+04</td> <td>   -1.707</td> <td> 0.088</td> <td>-3.51e+05</td> <td> 2.42e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1973.0]</th> <td>-8.893e+04</td> <td> 8.27e+04</td> <td>   -1.076</td> <td> 0.282</td> <td>-2.51e+05</td> <td> 7.31e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1974.0]</th> <td> -954.8271</td> <td> 1.18e+05</td> <td>   -0.008</td> <td> 0.994</td> <td>-2.32e+05</td> <td>  2.3e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1975.0]</th> <td> -2.01e+04</td> <td> 7.37e+04</td> <td>   -0.273</td> <td> 0.785</td> <td>-1.65e+05</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1976.0]</th> <td>-2.115e+05</td> <td> 1.65e+05</td> <td>   -1.282</td> <td> 0.200</td> <td>-5.35e+05</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1977.0]</th> <td> 3.225e+04</td> <td> 6.24e+04</td> <td>    0.517</td> <td> 0.605</td> <td>   -9e+04</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1978.0]</th> <td>-1.822e+05</td> <td> 9.67e+04</td> <td>   -1.884</td> <td> 0.060</td> <td>-3.72e+05</td> <td> 7315.391</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1979.0]</th> <td> 3897.6815</td> <td> 6.33e+04</td> <td>    0.062</td> <td> 0.951</td> <td> -1.2e+05</td> <td> 1.28e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1980.0]</th> <td> 5.464e+04</td> <td> 6.27e+04</td> <td>    0.872</td> <td> 0.383</td> <td>-6.82e+04</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1981.0]</th> <td>-3.348e+04</td> <td> 8.24e+04</td> <td>   -0.406</td> <td> 0.685</td> <td>-1.95e+05</td> <td> 1.28e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1982.0]</th> <td> 3.491e+04</td> <td> 6.25e+04</td> <td>    0.559</td> <td> 0.576</td> <td>-8.76e+04</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1983.0]</th> <td>-5738.1790</td> <td> 4.27e+04</td> <td>   -0.134</td> <td> 0.893</td> <td>-8.95e+04</td> <td>  7.8e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1984.0]</th> <td>-7.665e+04</td> <td> 4.28e+04</td> <td>   -1.791</td> <td> 0.073</td> <td>-1.61e+05</td> <td> 7254.095</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1985.0]</th> <td>-1.029e+05</td> <td> 4.42e+04</td> <td>   -2.327</td> <td> 0.020</td> <td>-1.89e+05</td> <td>-1.62e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1986.0]</th> <td>-2.057e+05</td> <td> 4.43e+04</td> <td>   -4.646</td> <td> 0.000</td> <td>-2.93e+05</td> <td>-1.19e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1987.0]</th> <td> 4.494e+05</td> <td> 4.43e+04</td> <td>   10.156</td> <td> 0.000</td> <td> 3.63e+05</td> <td> 5.36e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1988.0]</th> <td> -4.68e+04</td> <td> 4.99e+04</td> <td>   -0.937</td> <td> 0.349</td> <td>-1.45e+05</td> <td> 5.11e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1989.0]</th> <td> 2.613e+04</td> <td> 3.81e+04</td> <td>    0.686</td> <td> 0.493</td> <td>-4.86e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1990.0]</th> <td> -8.74e+04</td> <td> 3.62e+04</td> <td>   -2.417</td> <td> 0.016</td> <td>-1.58e+05</td> <td>-1.65e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1991.0]</th> <td>-7.945e+04</td> <td> 4.14e+04</td> <td>   -1.917</td> <td> 0.055</td> <td>-1.61e+05</td> <td> 1787.447</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1992.0]</th> <td>-1.316e+04</td> <td> 4.77e+04</td> <td>   -0.276</td> <td> 0.783</td> <td>-1.07e+05</td> <td> 8.04e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1993.0]</th> <td> 6.466e+04</td> <td> 4.78e+04</td> <td>    1.353</td> <td> 0.176</td> <td> -2.9e+04</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1994.0]</th> <td>  1.59e+05</td> <td> 4.43e+04</td> <td>    3.588</td> <td> 0.000</td> <td> 7.21e+04</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1995.0]</th> <td> 1.377e+05</td> <td> 4.77e+04</td> <td>    2.886</td> <td> 0.004</td> <td> 4.42e+04</td> <td> 2.31e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1996.0]</th> <td>-8821.4853</td> <td> 5.23e+04</td> <td>   -0.169</td> <td> 0.866</td> <td>-1.11e+05</td> <td> 9.36e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1997.0]</th> <td>-9.629e+04</td> <td> 5.23e+04</td> <td>   -1.841</td> <td> 0.066</td> <td>-1.99e+05</td> <td> 6255.923</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1998.0]</th> <td>-1.897e+04</td> <td> 4.14e+04</td> <td>   -0.458</td> <td> 0.647</td> <td>   -1e+05</td> <td> 6.22e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.1999.0]</th> <td> 1.264e+05</td> <td> 4.59e+04</td> <td>    2.756</td> <td> 0.006</td> <td> 3.65e+04</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2000.0]</th> <td>-5.066e+04</td> <td> 3.09e+04</td> <td>   -1.641</td> <td> 0.101</td> <td>-1.11e+05</td> <td> 9854.753</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2001.0]</th> <td> 2.028e+05</td> <td> 4.27e+04</td> <td>    4.747</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 2.87e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2002.0]</th> <td> 1.845e+05</td> <td> 4.02e+04</td> <td>    4.588</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2003.0]</th> <td> 7.542e+04</td> <td> 2.98e+04</td> <td>    2.531</td> <td> 0.011</td> <td>  1.7e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2004.0]</th> <td>-1781.1413</td> <td> 3.53e+04</td> <td>   -0.051</td> <td> 0.960</td> <td>-7.09e+04</td> <td> 6.73e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2005.0]</th> <td> 8.478e+04</td> <td> 3.09e+04</td> <td>    2.747</td> <td> 0.006</td> <td> 2.43e+04</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2006.0]</th> <td> 3.498e+04</td> <td>  3.7e+04</td> <td>    0.944</td> <td> 0.345</td> <td>-3.76e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2007.0]</th> <td> 4.727e+04</td> <td> 3.08e+04</td> <td>    1.536</td> <td> 0.124</td> <td> -1.3e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2008.0]</th> <td>  1.79e+05</td> <td> 4.42e+04</td> <td>    4.046</td> <td> 0.000</td> <td> 9.23e+04</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2009.0]</th> <td> 1.856e+05</td> <td> 3.61e+04</td> <td>    5.147</td> <td> 0.000</td> <td> 1.15e+05</td> <td> 2.56e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2010.0]</th> <td> 2.192e+05</td> <td> 4.27e+04</td> <td>    5.139</td> <td> 0.000</td> <td> 1.36e+05</td> <td> 3.03e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2011.0]</th> <td>-6.905e+04</td> <td> 5.51e+04</td> <td>   -1.253</td> <td> 0.210</td> <td>-1.77e+05</td> <td> 3.89e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2012.0]</th> <td> 5066.4086</td> <td> 5.84e+04</td> <td>    0.087</td> <td> 0.931</td> <td>-1.09e+05</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2013.0]</th> <td> 6.521e+04</td> <td> 2.98e+04</td> <td>    2.187</td> <td> 0.029</td> <td> 6760.894</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2014.0]</th> <td> 4.153e+04</td> <td> 1.95e+04</td> <td>    2.128</td> <td> 0.033</td> <td> 3269.731</td> <td> 7.98e+04</td>
</tr>
<tr>
  <th>C(yr_renovated)[T.2015.0]</th> <td>-5.634e+04</td> <td> 4.42e+04</td> <td>   -1.276</td> <td> 0.202</td> <td>-1.43e+05</td> <td> 3.02e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>       <td> 3.646e+04</td> <td>  1.5e+04</td> <td>    2.434</td> <td> 0.015</td> <td> 7095.040</td> <td> 6.58e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>       <td>-2.043e+04</td> <td> 1.33e+04</td> <td>   -1.530</td> <td> 0.126</td> <td>-4.66e+04</td> <td> 5735.374</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>       <td> 7.214e+05</td> <td> 2.49e+04</td> <td>   28.927</td> <td> 0.000</td> <td> 6.73e+05</td> <td>  7.7e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>       <td> 2.565e+05</td> <td> 2.65e+04</td> <td>    9.670</td> <td> 0.000</td> <td> 2.05e+05</td> <td> 3.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>       <td> 2.496e+05</td> <td> 2.17e+04</td> <td>   11.483</td> <td> 0.000</td> <td> 2.07e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>       <td> 2.164e+05</td> <td> 2.74e+04</td> <td>    7.887</td> <td> 0.000</td> <td> 1.63e+05</td> <td>  2.7e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>       <td> 2.435e+05</td> <td> 2.61e+04</td> <td>    9.313</td> <td> 0.000</td> <td> 1.92e+05</td> <td> 2.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>       <td> 1.191e+05</td> <td> 2.38e+04</td> <td>    5.009</td> <td> 0.000</td> <td> 7.25e+04</td> <td> 1.66e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>       <td> 5.912e+04</td> <td>  3.4e+04</td> <td>    1.737</td> <td> 0.082</td> <td>-7611.314</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>       <td>  1.32e+05</td> <td>  3.9e+04</td> <td>    3.388</td> <td> 0.001</td> <td> 5.56e+04</td> <td> 2.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>       <td> 7.743e+04</td> <td> 3.72e+04</td> <td>    2.084</td> <td> 0.037</td> <td> 4591.037</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>       <td>  8.36e+04</td> <td>  2.1e+04</td> <td>    3.979</td> <td> 0.000</td> <td> 4.24e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>       <td>-5.118e+04</td> <td> 1.24e+04</td> <td>   -4.138</td> <td> 0.000</td> <td>-7.54e+04</td> <td>-2.69e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>       <td> 1.752e+05</td> <td>  3.5e+04</td> <td>    5.012</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 2.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>       <td>  1.57e+05</td> <td> 2.24e+04</td> <td>    7.014</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>       <td> 4.622e+04</td> <td> 3.31e+04</td> <td>    1.396</td> <td> 0.163</td> <td>-1.87e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>       <td>  2.13e+05</td> <td> 2.56e+04</td> <td>    8.322</td> <td> 0.000</td> <td> 1.63e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>       <td> 8181.6905</td> <td> 1.48e+04</td> <td>    0.553</td> <td> 0.580</td> <td>-2.08e+04</td> <td> 3.72e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>       <td> 1.342e+04</td> <td> 1.54e+04</td> <td>    0.868</td> <td> 0.385</td> <td>-1.69e+04</td> <td> 4.37e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>       <td>-1.179e+04</td> <td> 1.78e+04</td> <td>   -0.662</td> <td> 0.508</td> <td>-4.67e+04</td> <td> 2.31e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>       <td> 3.149e+05</td> <td> 2.84e+04</td> <td>   11.095</td> <td> 0.000</td> <td> 2.59e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>       <td>  1.51e+05</td> <td> 3.05e+04</td> <td>    4.950</td> <td> 0.000</td> <td> 9.12e+04</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>       <td> 6.756e+04</td> <td> 1.69e+04</td> <td>    4.003</td> <td> 0.000</td> <td> 3.45e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>       <td> 1.283e+06</td> <td> 3.35e+04</td> <td>   38.345</td> <td> 0.000</td> <td> 1.22e+06</td> <td> 1.35e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>       <td> 4.817e+05</td> <td> 2.19e+04</td> <td>   21.968</td> <td> 0.000</td> <td> 4.39e+05</td> <td> 5.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>       <td> 2.594e+04</td> <td> 1.43e+04</td> <td>    1.817</td> <td> 0.069</td> <td>-2040.310</td> <td> 5.39e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>       <td>  1.74e+05</td> <td> 3.17e+04</td> <td>    5.488</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>       <td> 1.864e+05</td> <td>  2.9e+04</td> <td>    6.421</td> <td> 0.000</td> <td> 1.29e+05</td> <td> 2.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>       <td> 1.777e+05</td> <td> 3.13e+04</td> <td>    5.685</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>       <td> 2.959e+04</td> <td> 1.73e+04</td> <td>    1.710</td> <td> 0.087</td> <td>-4321.028</td> <td> 6.35e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>       <td> 7.795e+04</td> <td> 1.89e+04</td> <td>    4.126</td> <td> 0.000</td> <td> 4.09e+04</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>       <td> 2.426e+04</td> <td> 1.64e+04</td> <td>    1.479</td> <td> 0.139</td> <td>-7899.363</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>       <td> 7.557e+04</td> <td> 1.85e+04</td> <td>    4.081</td> <td> 0.000</td> <td> 3.93e+04</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>       <td> 1.302e+05</td> <td>  2.9e+04</td> <td>    4.495</td> <td> 0.000</td> <td> 7.34e+04</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>       <td>-8.992e+04</td> <td> 2.39e+04</td> <td>   -3.763</td> <td> 0.000</td> <td>-1.37e+05</td> <td>-4.31e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>       <td> 9.551e+04</td> <td>  3.4e+04</td> <td>    2.809</td> <td> 0.005</td> <td> 2.89e+04</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>       <td> 1.516e+05</td> <td> 2.75e+04</td> <td>    5.518</td> <td> 0.000</td> <td> 9.77e+04</td> <td> 2.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>       <td> 1.701e+05</td> <td> 2.63e+04</td> <td>    6.468</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 2.22e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>       <td> 7.755e+04</td> <td> 3.55e+04</td> <td>    2.184</td> <td> 0.029</td> <td> 7948.816</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>       <td> -2.57e+04</td> <td> 1.36e+04</td> <td>   -1.893</td> <td> 0.058</td> <td>-5.23e+04</td> <td>  908.955</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>       <td> 3.967e+05</td> <td> 2.89e+04</td> <td>   13.706</td> <td> 0.000</td> <td>  3.4e+05</td> <td> 4.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>       <td> 2.278e+05</td> <td> 2.74e+04</td> <td>    8.301</td> <td> 0.000</td> <td> 1.74e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>       <td> 3.584e+05</td> <td> 2.81e+04</td> <td>   12.760</td> <td> 0.000</td> <td> 3.03e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>       <td> 6.847e+04</td> <td> 2.02e+04</td> <td>    3.391</td> <td> 0.001</td> <td> 2.89e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>       <td> 2.342e+05</td> <td> 2.82e+04</td> <td>    8.300</td> <td> 0.000</td> <td> 1.79e+05</td> <td> 2.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>       <td> 4.622e+04</td> <td> 2.22e+04</td> <td>    2.083</td> <td> 0.037</td> <td> 2731.412</td> <td> 8.97e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>       <td> 3.931e+05</td> <td> 2.89e+04</td> <td>   13.624</td> <td> 0.000</td> <td> 3.37e+05</td> <td>  4.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>       <td> 5.105e+05</td> <td> 2.57e+04</td> <td>   19.835</td> <td> 0.000</td> <td>  4.6e+05</td> <td> 5.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>       <td> 2.252e+05</td> <td> 2.79e+04</td> <td>    8.062</td> <td> 0.000</td> <td>  1.7e+05</td> <td>  2.8e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>       <td> 2.057e+05</td> <td> 2.26e+04</td> <td>    9.112</td> <td> 0.000</td> <td> 1.61e+05</td> <td>  2.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>       <td> 1.979e+05</td> <td> 2.83e+04</td> <td>    6.995</td> <td> 0.000</td> <td> 1.42e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>       <td> 1.141e+05</td> <td> 1.97e+04</td> <td>    5.792</td> <td> 0.000</td> <td> 7.55e+04</td> <td> 1.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>       <td> 3.754e+05</td> <td> 2.73e+04</td> <td>   13.731</td> <td> 0.000</td> <td> 3.22e+05</td> <td> 4.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>       <td> 2.577e+05</td> <td> 2.45e+04</td> <td>   10.525</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 3.06e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>       <td> 1.102e+05</td> <td> 3.02e+04</td> <td>    3.648</td> <td> 0.000</td> <td>  5.1e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>       <td> 1.118e+05</td> <td> 2.07e+04</td> <td>    5.390</td> <td> 0.000</td> <td> 7.11e+04</td> <td> 1.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>       <td> 6.376e+04</td> <td> 3.12e+04</td> <td>    2.045</td> <td> 0.041</td> <td> 2636.234</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>       <td> 1.756e+05</td> <td> 2.12e+04</td> <td>    8.294</td> <td> 0.000</td> <td> 1.34e+05</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>       <td> 2.035e+05</td> <td> 2.27e+04</td> <td>    8.960</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 2.48e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>       <td> 5.865e+04</td> <td>  1.9e+04</td> <td>    3.087</td> <td> 0.002</td> <td> 2.14e+04</td> <td> 9.59e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>       <td> 3.345e+04</td> <td> 2.55e+04</td> <td>    1.314</td> <td> 0.189</td> <td>-1.65e+04</td> <td> 8.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>       <td>  5.96e+04</td> <td> 3.25e+04</td> <td>    1.835</td> <td> 0.067</td> <td>-4074.016</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>       <td> 1.601e+04</td> <td> 1.73e+04</td> <td>    0.925</td> <td> 0.355</td> <td>-1.79e+04</td> <td> 4.99e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>       <td> 1.661e+04</td> <td> 1.83e+04</td> <td>    0.909</td> <td> 0.363</td> <td>-1.92e+04</td> <td> 5.24e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>       <td> 1.362e+05</td> <td> 3.25e+04</td> <td>    4.189</td> <td> 0.000</td> <td> 7.25e+04</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>       <td> 2277.0673</td> <td> 1.89e+04</td> <td>    0.121</td> <td> 0.904</td> <td>-3.47e+04</td> <td> 3.93e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>       <td> 8578.8637</td> <td> 1.91e+04</td> <td>    0.448</td> <td> 0.654</td> <td> -2.9e+04</td> <td> 4.61e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>       <td>-1143.8310</td> <td> 1.45e+04</td> <td>   -0.079</td> <td> 0.937</td> <td>-2.95e+04</td> <td> 2.72e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>       <td> 2.662e+05</td> <td> 2.68e+04</td> <td>    9.937</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 3.19e+05</td>
</tr>
<tr>
  <th>bedrooms</th>                  <td>-3.232e+04</td> <td> 1711.610</td> <td>  -18.882</td> <td> 0.000</td> <td>-3.57e+04</td> <td> -2.9e+04</td>
</tr>
<tr>
  <th>bathrooms</th>                 <td> 2.168e+04</td> <td> 2777.151</td> <td>    7.807</td> <td> 0.000</td> <td> 1.62e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>scaled_sqft_living</th>        <td> 2.273e+06</td> <td> 3.17e+04</td> <td>   71.783</td> <td> 0.000</td> <td> 2.21e+06</td> <td> 2.34e+06</td>
</tr>
<tr>
  <th>scaled_sqft_lot</th>           <td> 2.209e+05</td> <td> 5.99e+04</td> <td>    3.688</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>floors</th>                    <td>-2.405e+04</td> <td> 3229.987</td> <td>   -7.444</td> <td> 0.000</td> <td>-3.04e+04</td> <td>-1.77e+04</td>
</tr>
<tr>
  <th>grade</th>                     <td> 7.527e+04</td> <td> 1829.495</td> <td>   41.145</td> <td> 0.000</td> <td> 7.17e+04</td> <td> 7.89e+04</td>
</tr>
<tr>
  <th>scaled_lat</th>                <td>  9.67e+04</td> <td> 4.23e+04</td> <td>    2.288</td> <td> 0.022</td> <td> 1.39e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>scaled_long</th>               <td>-2.304e+05</td> <td> 5.91e+04</td> <td>   -3.901</td> <td> 0.000</td> <td>-3.46e+05</td> <td>-1.15e+05</td>
</tr>
<tr>
  <th>scaled_sqft_lot15</th>         <td> 3.881e+04</td> <td> 1.51e+04</td> <td>    2.572</td> <td> 0.010</td> <td> 9235.053</td> <td> 6.84e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>18978.217</td> <th>  Durbin-Watson:     </th>  <td>   1.989</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>2734243.685</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.810</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>58.299</td>   <th>  Cond. No.          </th>  <td>1.56e+03</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.56e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# The R2 value is still at 0.804 so that's good. 
# The bedrooms coeff is negative so we may need to scale that as well to get the magnitude of units closer. 
# The scaled_sqft_living coeff went from 194.6 to 2.273^6
# The scaled_sqft_lot coeff went from 0.1898 to 2.209^5
# The scaled_lat coeff went from 1.555e+05 to 9.67e+04
# The scaled_long coeff went from -1.922e+05 to -2.304e+05
# The sqft_lot15 coeff went from 0.4128 to 3.881e+04

# Our skew is still 3.8. 
# Kurtosis is still 58.3.
# The condition number is down to 1.56^3 from 1.82^8 in the last model. 
```

### Look at heat map again with new numbers. 


```python
fig, axes = plt.subplots(figsize=(13,13))
sns.heatmap(df_filtered1.corr().round(3), center=0, square=True, ax=axes, annot=True);
axes.set_ylim(len(df_filtered1.corr()),-0.5,+0.5)
```




    (20, -0.5)




![png](output_158_1.png)



```python
# it seems like grade is somewhat correlated so maybe I could drop that column, 
# or I can try scaling it to see if that changes anything. 
```

### New model with only specific coeffs to compare.


```python
# create model using best coeffs list of predictors by typing them all out. 
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'scaled_sqft_living', 'scaled_sqft_lot',
        'C(condition)', 'grade',]
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = smf.ols(formula=formula, data=df_filtered1).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.559</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.559</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   2965.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 19 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>18:53:11</td>     <th>  Log-Likelihood:    </th> <td>-2.9114e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21059</td>      <th>  AIC:               </th>  <td>5.823e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21049</td>      <th>  BIC:               </th>  <td>5.824e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>          <td>-3.249e+05</td> <td> 4.82e+04</td> <td>   -6.742</td> <td> 0.000</td> <td>-4.19e+05</td> <td> -2.3e+05</td>
</tr>
<tr>
  <th>C(condition)[T.2]</th>  <td>  -8.4e+04</td> <td> 5.02e+04</td> <td>   -1.674</td> <td> 0.094</td> <td>-1.82e+05</td> <td> 1.44e+04</td>
</tr>
<tr>
  <th>C(condition)[T.3]</th>  <td>-1.357e+05</td> <td> 4.63e+04</td> <td>   -2.929</td> <td> 0.003</td> <td>-2.27e+05</td> <td>-4.49e+04</td>
</tr>
<tr>
  <th>C(condition)[T.4]</th>  <td>-6.965e+04</td> <td> 4.64e+04</td> <td>   -1.502</td> <td> 0.133</td> <td>-1.61e+05</td> <td> 2.12e+04</td>
</tr>
<tr>
  <th>C(condition)[T.5]</th>  <td> 1.597e+04</td> <td> 4.66e+04</td> <td>    0.342</td> <td> 0.732</td> <td>-7.54e+04</td> <td> 1.07e+05</td>
</tr>
<tr>
  <th>bedrooms</th>           <td>-5.142e+04</td> <td> 2415.722</td> <td>  -21.285</td> <td> 0.000</td> <td>-5.62e+04</td> <td>-4.67e+04</td>
</tr>
<tr>
  <th>bathrooms</th>          <td>-1.714e+04</td> <td> 3530.467</td> <td>   -4.855</td> <td> 0.000</td> <td>-2.41e+04</td> <td>-1.02e+04</td>
</tr>
<tr>
  <th>scaled_sqft_living</th> <td> 2.707e+06</td> <td> 4.35e+04</td> <td>   62.265</td> <td> 0.000</td> <td> 2.62e+06</td> <td> 2.79e+06</td>
</tr>
<tr>
  <th>scaled_sqft_lot</th>    <td>-5.415e+05</td> <td> 7.45e+04</td> <td>   -7.270</td> <td> 0.000</td> <td>-6.88e+05</td> <td>-3.96e+05</td>
</tr>
<tr>
  <th>grade</th>              <td> 1.033e+05</td> <td> 2364.246</td> <td>   43.680</td> <td> 0.000</td> <td> 9.86e+04</td> <td> 1.08e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>15984.277</td> <th>  Durbin-Watson:     </th>  <td>   1.991</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>882814.966</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.151</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>34.087</td>   <th>  Cond. No.          </th>  <td>    544.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# that whole model is messed up now. Pulling up qq plots.

plt.style.use('ggplot')

f = 'price~bedrooms'
f2 = 'price~bathrooms'
f3 = 'price~scaled_sqft_living'
f4 = 'price~scaled_sqft_lot'
# f5 = 'price~C(condition)'
f6 = 'price~grade'
model = smf.ols(formula=f, data=df_filtered1).fit()
model2 = smf.ols(formula=f2, data=df_filtered1).fit()
model3 = smf.ols(formula=f3, data=df_filtered1).fit()
model4 = smf.ols(formula=f4, data=df_filtered1).fit()
# model5 = smf.ols(formula=f5, data=data).fit()
model6 = smf.ols(formula=f6, data=df_filtered1).fit()


resid1 = model.resid
resid2 = model2.resid
resid3 = model3.resid
resid4 = model4.resid
# resid5 = model5.resid
resid6 = model6.resid

fig = sm.graphics.qqplot(resid1, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid2, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid3, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid4, dist=stats.norm, line='45', fit=True)
# fig = sm.graphics.qqplot(resid5, dist=stats.norm, line='45', fit=True)
fig = sm.graphics.qqplot(resid6, dist=stats.norm, line='45', fit=True)
```


![png](output_162_0.png)



![png](output_162_1.png)



![png](output_162_2.png)



![png](output_162_3.png)



![png](output_162_4.png)



```python
# all of my variables are messed up and have outliers falling way off of the line. No clue how to fix this. Filter the price ?
```

# iNTERPRET


```python
df_filtered1.shape
```




    (21059, 20)




```python
df_filtered1.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.105900e+04</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>2.105900e+04</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
      <td>21059.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.392477e+05</td>
      <td>3.373854</td>
      <td>2.114072</td>
      <td>2071.458521</td>
      <td>1.195228e+04</td>
      <td>1.495133</td>
      <td>0.006790</td>
      <td>3.412033</td>
      <td>7.654969</td>
      <td>1970.897621</td>
      <td>69.097251</td>
      <td>98078.450639</td>
      <td>47.561666</td>
      <td>-122.216805</td>
      <td>9797.450401</td>
      <td>0.145673</td>
      <td>0.009819</td>
      <td>0.652672</td>
      <td>0.247869</td>
      <td>0.097290</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.679167e+05</td>
      <td>0.903651</td>
      <td>0.765593</td>
      <td>905.122712</td>
      <td>2.686565e+04</td>
      <td>0.540648</td>
      <td>0.082126</td>
      <td>0.650613</td>
      <td>1.164359</td>
      <td>29.465601</td>
      <td>364.913382</td>
      <td>53.636967</td>
      <td>0.137634</td>
      <td>0.138271</td>
      <td>10169.465078</td>
      <td>0.077493</td>
      <td>0.023075</td>
      <td>0.221384</td>
      <td>0.115322</td>
      <td>0.108172</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.514000</td>
      <td>651.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.200000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1420.000000</td>
      <td>5.001000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.474700</td>
      <td>-122.329000</td>
      <td>5075.000000</td>
      <td>0.089897</td>
      <td>0.003849</td>
      <td>0.512788</td>
      <td>0.154295</td>
      <td>0.047058</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.540000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.573600</td>
      <td>-122.234000</td>
      <td>7560.000000</td>
      <td>0.131849</td>
      <td>0.006030</td>
      <td>0.671867</td>
      <td>0.233528</td>
      <td>0.073491</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.430000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2540.000000</td>
      <td>1.040000e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678550</td>
      <td>-122.129000</td>
      <td>9930.000000</td>
      <td>0.185788</td>
      <td>0.008486</td>
      <td>0.840679</td>
      <td>0.321101</td>
      <td>0.098700</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>11.000000</td>
      <td>8.000000</td>
      <td>12050.000000</td>
      <td>1.164794e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>94663.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_basic.shape
```




    (21420, 20)




```python
df_basic.describe()
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
      <th>z_score_bedrooms</th>
      <th>z_score_bathrooms</th>
      <th>z_score_sqft_living</th>
      <th>z_score_sqft_lot</th>
      <th>z_score_sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.142000e+04</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>2.142000e+04</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.00000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.407393e+05</td>
      <td>3.372549</td>
      <td>2.118429</td>
      <td>2083.132633</td>
      <td>1.512804e+04</td>
      <td>1.495985</td>
      <td>0.006816</td>
      <td>3.410784</td>
      <td>7.662792</td>
      <td>1971.092997</td>
      <td>68.956723</td>
      <td>98077.87437</td>
      <td>47.560197</td>
      <td>-122.213784</td>
      <td>12775.718161</td>
      <td>0.810073</td>
      <td>0.798798</td>
      <td>0.760465</td>
      <td>0.334145</td>
      <td>0.370691</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.679311e+05</td>
      <td>0.902995</td>
      <td>0.768720</td>
      <td>918.808412</td>
      <td>4.153080e+04</td>
      <td>0.540081</td>
      <td>0.082280</td>
      <td>0.650035</td>
      <td>1.171971</td>
      <td>29.387141</td>
      <td>364.552298</td>
      <td>53.47748</td>
      <td>0.138589</td>
      <td>0.140791</td>
      <td>27345.621867</td>
      <td>0.586343</td>
      <td>0.601613</td>
      <td>0.649394</td>
      <td>0.942544</td>
      <td>0.928778</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.00000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>651.000000</td>
      <td>0.412580</td>
      <td>0.154064</td>
      <td>0.002032</td>
      <td>0.000097</td>
      <td>0.000120</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.225000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1952.000000</td>
      <td>0.000000</td>
      <td>98033.00000</td>
      <td>47.471200</td>
      <td>-122.328000</td>
      <td>5100.000000</td>
      <td>0.412580</td>
      <td>0.479287</td>
      <td>0.312224</td>
      <td>0.143946</td>
      <td>0.146488</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1920.000000</td>
      <td>7.614000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.00000</td>
      <td>47.572100</td>
      <td>-122.230000</td>
      <td>7620.000000</td>
      <td>0.694872</td>
      <td>0.496383</td>
      <td>0.645561</td>
      <td>0.199749</td>
      <td>0.218531</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.069050e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98117.00000</td>
      <td>47.678100</td>
      <td>-122.125000</td>
      <td>10086.250000</td>
      <td>0.694872</td>
      <td>1.454958</td>
      <td>1.037382</td>
      <td>0.263571</td>
      <td>0.307340</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>11.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.00000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>871200.000000</td>
      <td>8.447032</td>
      <td>7.651302</td>
      <td>12.469558</td>
      <td>39.398935</td>
      <td>31.392386</td>
    </tr>
  </tbody>
</table>
</div>



### bedroom vs bathroom


```python
# it looks like my data was filtered above but was never removed from the df so I've been running my model 
# with all the outliers from original df. 
```


```python
# bedrooms	-32320
# bathrooms	21680
# according to my previous model the bathrooms have a bigger coeff and more positive relationship with price.
df_filtered1.plot(kind='scatter', x='bedrooms', y='price', alpha=0.4, color='b')
df_filtered1.plot(kind='scatter', x='bathrooms', y='price', alpha=0.4, color='b');
```


![png](output_171_0.png)



![png](output_171_1.png)


### sqft living vs sqft lot


```python
# scaled_sqft_living	2273000
# scaled_sqft_lot	220900
# according to model my sqft living area is more significant
df_filtered1.plot(kind='scatter', x='sqft_living', y='price', alpha=0.4, color='b')
df_filtered1.plot(kind='scatter', x='sqft_lot', y='price', alpha=0.4, color='b');
```


![png](output_173_0.png)



![png](output_173_1.png)


### condition vs grade


```python
# grade	75270
# C(condition)[T.2]	54930
# C(condition)[T.3]	47460
# C(condition)[T.4]	69170
# C(condition)[T.5]	103100
# grade is more significant unless your house is condition level 5. 
df_filtered1.plot(kind='scatter', x='condition', y='price', alpha=0.4, color='b')
df_filtered1.plot(kind='scatter', x='grade', y='price', alpha=0.4, color='b');
```


![png](output_175_0.png)



![png](output_175_1.png)



```python

```

# CONCLUSIONS & RECOMMENDATIONS

> Best predictors for house pricing:
    - # of bathrooms (recommend adding a bathroom to your home)
    - square footage of living area (recommend finding most sqft you can afford)
    - grade of house based on King County grading system (look for best grade rating for quality. higher grade 
        is also usually paired with higher sqft area too.)



```python

```
