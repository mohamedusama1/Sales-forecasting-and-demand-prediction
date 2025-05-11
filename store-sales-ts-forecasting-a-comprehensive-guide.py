#!/usr/bin/env python
# coding: utf-8

# There are **54 stores** and **33 prodcut families** in the data. The time serie starts from **2013-01-01** and finishes in **2017-08-31**. However, you know that Kaggle gives us splitted two data as train and test. The dates in the test data are for the **15 days** after the last date in the training data. Date range in the test data will be very important to us while we are defining a cross-validation strategy and creating new features.
# 
# ***Our main mission in this competition is, predicting sales for each product family and store combinations.***
# 
# There are 6 data that we will study on them step by step.
# 1. *Train*
# 2. *Test*
# 3. *Store*
# 4. *Transactions* 
# 5. *Holidays and Events*
# 6. *Daily Oil Price*
# 
# **<code>The train data</code>** contains time series of the stores and the product families combination. The sales column gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).The onpromotion column gives the total number of items in a product family that were being promoted at a store at a given date.
# 
# **<code>Stores data</code>** gives some information about stores such as city, state, type, cluster.
# 
# **<code>Transaction data</code>** is highly correlated with train's sales column. You can understand the sales patterns of the stores.
# 
# **<code>Holidays and events data</code>** is a meta data. This data is quite valuable to understand past sales, trend and seasonality components. However, it needs to be arranged. You are going to find a comprehensive data manipulation for this data. That part will be one of the most important chapter in this notebook.
# 
# **<code>Daily Oil Price data</code>** is another data which will help us. Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices. That's why, it will help us to understand which product families affected in positive or negative way by oil price.
# 
# #### When you look at the data description, you will see "Additional Notes". These notes may be significant to catch some patterns or anomalies. I'm sharing them with you to remember.
# - Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
# - A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
# 
# <center><h2> Let's start! </h2> </center>
# 
# <center><img src="https://media.istockphoto.com/photos/flag-and-church-in-guayaquil-picture-id481766414?k=20&m=481766414&s=612x612&w=0&h=s8CFr9trtS6Dc3XlecsV1yTzw4FrUQR97ScDbym33jc=" style="width:70%;height:10%;"></center>
# 

# # 1. Packages
# 
# You can find the packages below what I used.

# In[158]:


# BASE
# ------------------------------------------------------
import numpy as np
import pandas as pd
import os
import gc
import warnings

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# CONFIGURATIONS
# ------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')


# # 2. Importing Data

# In[159]:


# Import
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
stores = pd.read_csv("./stores.csv")
transactions = pd.read_csv("./transactions.csv").sort_values(["store_nbr", "date"])


# Datetime
train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

train.head()


# # 3. Transactions
# 
# **Let's start with the transaction data**

# In[160]:


transactions.head(10)


# This feature is highly correlated with sales but first, you are supposed to sum the sales feature to find relationship. Transactions means how many people came to the store or how many invoices created in a day.
# 
# Sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
# 
# That's why, transactions will be one of the relevant features in the model. In the following sections, we will generate new features by using transactions.

# In[161]:


temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how = "left")
print("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))
px.line(transactions.sort_values(["store_nbr", "date"]), x='date', y='transactions', color='store_nbr',title = "Transactions" )


# There is a stable pattern in Transaction. All months are similar except December from 2013 to 2017 by boxplot. In addition, we've just seen same pattern for each store in previous plot. Store sales had always increased at the end of the year.

# In[162]:


a = transactions.copy()
a["year"] = a.date.dt.year
a["month"] = a.date.dt.month
px.box(a, x="year", y="transactions" , color = "month", title = "Transactions")


# **Let's take a look at transactions by using monthly average sales!**
# 
#  We've just learned a pattern what increases sales. It was the end of the year. We can see that transactions increase in spring and decrease after spring.

# In[163]:


a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
a["year"] = a.date.dt.year
px.line(a, x='date', y='transactions', color='year',title = "Monthly Average Transactions" )


# When we look at their relationship, we can see that there is a highly correlation between total sales and transactions also. 

# In[164]:


px.scatter(temp, x = "transactions", y = "sales", trendline = "ols", trendline_color_override = "red")


# The days of week is very important for shopping. It shows us a great pattern. Stores make more transactions at weekends. Almost, the patterns are same from 2013 to 2017 and Saturday is the most important day for shopping.

# In[165]:


a = transactions.copy()
a["year"] = a.date.dt.year
a["dayofweek"] = a.date.dt.dayofweek+1
a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
px.line(a, x="dayofweek", y="transactions" , color = "year", title = "Transactions")


# # 4. Oil Price
# 
# 
# The economy is one of the biggest problem for the governments and people. It affects all of things in a good or bad way. In our case, Ecuador is an oil-dependent country. Changing oil prices in Ecuador will cause a variance in the model. I researched Ecuador's economy to be able to understand much better and I found an article from IMF. You are supposed to read it if you want to make better models by using oil data.
# 
# - https://www.imf.org/en/News/Articles/2019/03/20/NA032119-Ecuador-New-Economic-Plan-Explained
# 
# <br>
# 
# <center><img src="https://github.com/EkremBayar/Kaggle/blob/main/Images/imf_sf.PNG?raw=true" style="width:50%;height:10%;"></center>
# 
# <br>
# 
# There are some missing data points in the daily oil data as you can see below. You can treat the data by using various imputation methods. However, I chose a simple solution for that. Linear Interpolation is suitable for this time serie. You can see the trend and predict missing data points, when you look at a time serie plot of oil price.

# In[166]:


# 📥 Import 
oil = pd.read_csv("./oil.csv")
oil["date"] = pd.to_datetime(oil.date)

# 📅 Resample (to daily frequency)
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()

# ⚡ Handle missing and zero values
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])

# 🔄 Interpolate inside gaps
oil["dcoilwtico_interpolated"] = oil["dcoilwtico"].interpolate()

# 🔙 Backfill beginning NaNs
oil["dcoilwtico_interpolated"] = oil["dcoilwtico_interpolated"].fillna(method='bfill')

# 🎨 Plot
p = oil.melt(id_vars=['date'], value_vars=["dcoilwtico_interpolated"], var_name='Legend')
px.line(
    p.sort_values(["Legend", "date"], ascending=[False, True]),
    x='date', y='value', color='Legend', title="Daily Oil Price"
)


# **I just said, "Ecuador is a oil-dependent country" but is it true? Can we really see that from the data by looking at?**
# 
# First of all, let's look at the correlations for sales and transactions. The correlation values are not strong but the sign of sales is negative. Maybe, we can catch a clue. Logically, if daily oil price is high, we expect that the Ecuador's economy is bad and it means the price of product increases and sales decreases. There is a negative relationship here.  

# In[167]:


temp = pd.merge(temp, oil, how = "left")
print("Correlation with Daily Oil Prices")
print(temp.drop(["store_nbr", "dcoilwtico"], axis = 1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")


fig, axes = plt.subplots(1, 2, figsize = (15,5))
temp.plot.scatter(x = "dcoilwtico_interpolated", y = "transactions", ax=axes[0])
temp.plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[1], color = "r")
axes[0].set_title('Daily oil price & Transactions', fontsize = 15)
axes[1].set_title('Daily Oil Price & Sales', fontsize = 15);


# You should never decide what you will do by looking at a graph or result! You are supposed to change your view and define new hypotheses.
# 
# We would have been wrong if we had looked at some simple outputs just like above and we had said that there is no relationship with oil prices and let's not use oil price data.
# 
# All right! We are aware of analyzing deeply now. Let's draw a scatter plot but let's pay attention for product families this time. All of the plots almost contains same pattern. When daily oil price is under about 70, there are more sales in the data. There are 2 cluster here. They are over 70 and under 70. It seems pretty understandable actually. 
# 
# We are in a good way I think. What do you think? Just now, we couldn't see a pattern for daily oil price, but now we extracted a new pattern from it.

# In[168]:


a = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis = 1), how = "left")
c = a.groupby("family").corr("spearman").reset_index()
c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

fig, axes = plt.subplots(7, 5, figsize = (20,20))
for i, fam in enumerate(c.family):
    if i < 6:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[0, i-1])
        axes[0, i-1].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[0, i-1].axvline(x=70, color='r', linestyle='--')
    if i >= 6 and i<11:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[1, i-6])
        axes[1, i-6].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[1, i-6].axvline(x=70, color='r', linestyle='--')
    if i >= 11 and i<16:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[2, i-11])
        axes[2, i-11].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[2, i-11].axvline(x=70, color='r', linestyle='--')
    if i >= 16 and i<21:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[3, i-16])
        axes[3, i-16].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[3, i-16].axvline(x=70, color='r', linestyle='--')
    if i >= 21 and i<26:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[4, i-21])
        axes[4, i-21].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[4, i-21].axvline(x=70, color='r', linestyle='--')
    if i >= 26 and i < 31:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[5, i-26])
        axes[5, i-26].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[5, i-26].axvline(x=70, color='r', linestyle='--')
    if i >= 31 :
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[6, i-31])
        axes[6, i-31].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[6, i-31].axvline(x=70, color='r', linestyle='--')


plt.tight_layout(pad=5)
plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize = 20);
plt.show()



# # 5. Sales
# 
# Our main objective is, predicting store sales for each product family. For this reason, sales column should be examined more seriously. We need to learn everthing such as seasonality, trends, anomalies, similarities with other time series and so on.

# Most of the stores are similar to each other, when we examine them with correlation matrix. Some stores, such as 20, 21, 22, and 52 may be a little different.

# In[169]:


a = train[["store_nbr", "sales"]]
a["ind"] = 1
a["ind"] = a.groupby("store_nbr").ind.cumsum().values
a = pd.pivot(a, index = "ind", columns = "store_nbr", values = "sales").corr()
mask = np.triu(a.corr())
plt.figure(figsize=(20, 20))
sns.heatmap(a,
        annot=True,
        fmt='.1f',
        cmap='coolwarm',
        square=True,
        mask=mask,
        linewidths=1,
        cbar=False)
plt.title("Correlations among stores",fontsize = 20)
plt.show()


# There is a graph that shows us daily total sales below.

# In[170]:


a = train.set_index("date").groupby("store_nbr").resample("D").sales.sum().reset_index()
px.line(a, x = "date", y= "sales", color = "store_nbr", title = "Daily total sales of the stores")


# I realized some unnecessary rows in the data while I was looking at the time serie of the stores one by one. If you select the stores from above, some of them have no sales at the beginning of 2013. You can see them, if you look at the those stores 20, 21, 22, 29, 36, 42, 52 and 53. I decided to remove those rows before the stores opened. In the following codes, we will get rid of them.

# In[171]:


print(train.shape)
train = train[~((train.store_nbr == 52) & (train.date < "2017-04-20"))]
train = train[~((train.store_nbr == 22) & (train.date < "2015-10-09"))]
train = train[~((train.store_nbr == 42) & (train.date < "2015-08-21"))]
train = train[~((train.store_nbr == 21) & (train.date < "2015-07-24"))]
train = train[~((train.store_nbr == 29) & (train.date < "2015-03-20"))]
train = train[~((train.store_nbr == 20) & (train.date < "2015-02-13"))]
train = train[~((train.store_nbr == 53) & (train.date < "2014-05-29"))]
train = train[~((train.store_nbr == 36) & (train.date < "2013-05-09"))]
train.shape


# ## Zero Forecasting
# 
# Some stores don't sell some product families. In the following code, you can see which products aren't sold in which stores. It isn't difficult to forecast them next 15 days. Their forecasts must be 0 next 15 days.
# 
# I will remove them from the data and create a new data frame for product families which never sell. Then, when we are at submission part, I will combine that data frame with our predictions.

# In[172]:


c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family","store_nbr"])
c = c[c.sales == 0]
c


# In[173]:


print(train.shape)
# Anti Join
outer_join = train.merge(c[c.sales == 0].drop("sales",axis = 1), how = 'outer', indicator = True)
train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
del outer_join
gc.collect()
print(train.shape)


# In[174]:


zero_prediction = []
for i in range(0,len(c)):
    zero_prediction.append(
        pd.DataFrame({
            "date":pd.date_range("2017-08-16", "2017-08-31").tolist(),
            "store_nbr":c.store_nbr.iloc[i],
            "family":c.family.iloc[i],
            "sales":0
        })
    )
zero_prediction = pd.concat(zero_prediction)
del c
gc.collect()
zero_prediction


# ## Are The Product Families Active or Passive?
# 
# Some products can sell rarely in the stores. When I worked on a product supply demand for restuarants project at my previous job, some products were passive if they never bought in the last two months. I want to apply this domain knowledge here and I will look on the last 60 days.
# 
# However, some product families depends on seasonality. Some of them might not active on the last 60 days but it doesn't mean it is passive.

# In[175]:


c = train.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()
c[c.sales == 0]


# As you can see below, these examples are too rare and also the sales are low. I'm open your suggestions for these families. I won't do anything for now but, you would like to improve your model you can focus on that.
# 
# But still, I want to use that knowledge whether it is simple and I will create a new feature. It shows that the product family is active or not.

# In[176]:


fig, ax = plt.subplots(1,5, figsize = (20,4))
train[(train.store_nbr == 10) & (train.family == "LAWN AND GARDEN")].set_index("date").sales.plot(ax = ax[0], title = "STORE 10 - LAWN AND GARDEN")
train[(train.store_nbr == 36) & (train.family == "LADIESWEAR")].set_index("date").sales.plot(ax = ax[1], title = "STORE 36 - LADIESWEAR")
train[(train.store_nbr == 6) & (train.family == "SCHOOL AND OFFICE SUPPLIES")].set_index("date").sales.plot(ax = ax[2], title = "STORE 6 - SCHOOL AND OFFICE SUPPLIES")
train[(train.store_nbr == 14) & (train.family == "BABY CARE")].set_index("date").sales.plot(ax = ax[3], title = "STORE 14 - BABY CARE")
train[(train.store_nbr == 53) & (train.family == "BOOKS")].set_index("date").sales.plot(ax = ax[4], title = "STORE 43 - BOOKS")
plt.show()


# We can catch the trends, seasonality and anomalies for families.

# In[177]:


a = train.set_index("date").groupby("family").resample("D").sales.sum().reset_index()
px.line(a, x = "date", y= "sales", color = "family", title = "Daily total sales of the family")


# We are working with the stores. Well, there are plenty of products in the stores and we need to know which product family sells much more? Let's make a barplot to see that.
# 
# The graph shows us GROCERY I and BEVERAGES are the top selling families.

# In[178]:


a = train.groupby("family").sales.mean().sort_values(ascending = False).reset_index()
px.bar(a, y = "family", x="sales", color = "family", title = "Which product family preferred more?")


# #### Does onpromotion column cause a data leakage problem?
# 
# It is really a good question. The Data Leakage is one of the biggest problem when we will fit a model. There is a great discussion from [Nesterenko Marina](https://www.kaggle.com/nesterenkomarina) [@nesterenkomarina](https://www.kaggle.com/nesterenkomarina). You should look at it before fitting a model.
# 
# - https://www.kaggle.com/c/store-sales-time-series-forecasting/discussion/277067

# In[179]:


# print("Spearman Correlation between Sales and Onpromotion: {:,.4f}".format(train.corr("spearman").sales.loc["onpromotion"]))


# 
# <center><img src="https://github.com/EkremBayar/Kaggle/blob/main/Images/kr.PNG?raw=true
# " style="width:90%;height:10%;"></center>

# How different can stores be from each other? I couldn't find a major pattern among the stores actually. But I only looked at a single plot. There may be some latent patterns. 

# In[180]:


d = pd.merge(train, stores)
d["store_nbr"] = d["store_nbr"].astype("int8")
d["year"] = d.date.dt.year
px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x = "year", y = "sales", color = "city")


# # 6. Holidays and Events
# 
# What a mess! Probably, you are confused due to the holidays and events data. It contains a lot of information inside but, don't worry. You just need to take a breathe and think! It is a meta-data so you have to split it logically and make the data useful.
# 
# What are our problems?
# - Some national holidays have been transferred.
# - There might be a few holidays in one day. When we merged all of data, number of rows might increase. We don't want duplicates.
# - What is the scope of holidays? It can be regional or national or local. You need to split them by the scope.
# - Work day issue
# - Some specific events
# - Creating new features etc.
# 
# 
# End of the section, they won't be a problem anymore!

# In[181]:


import numpy as np
import pandas as pd

holidays = pd.read_csv("./holidays_events.csv")
holidays["date"] = pd.to_datetime(holidays.date)

# --- Helper functions for holiday processing ---
def create_holiday_features(df, holidays):
    """
    Create holiday-related features from the holidays data

    Parameters:
    -----------
    df : pandas.DataFrame
        Main dataframe
    holidays : pandas.DataFrame
        Holidays dataframe with required columns

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with added holiday features and work_day dataframe
    """
    # Parse date if not already done
    if not pd.api.types.is_datetime64_dtype(holidays['date']):
        holidays['date'] = pd.to_datetime(holidays['date'])

    # Ensure description is string type
    holidays['description'] = holidays['description'].fillna('').astype(str)

    # Fix "transferred" column if it's not a boolean
    if not pd.api.types.is_bool_dtype(holidays['transferred']):
        holidays['transferred'] = holidays['transferred'].astype(bool)

    # Transferred Holidays
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1).reset_index(drop=True)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis=1).reset_index(drop=True)

    # Check if there are any rows before trying to concatenate
    if len(tr1) > 0 and len(tr2) > 0:
        tr = pd.concat([tr1, tr2], axis=1)
        tr = tr.iloc[:, [5,1,2,3,4]] if tr.shape[1] > 5 else tr  # Ensure proper indexing
    else:
        # Create an empty DataFrame with the same columns
        tr = pd.DataFrame(columns=holidays.columns)

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)

    # Only concatenate if tr is not empty
    if not tr.empty:
        holidays = pd.concat([holidays, tr], axis=0).reset_index(drop=True)

    # Additional Holidays - safely apply string operations
    holidays["description"] = holidays["description"].str.replace("-", "", regex=False).str.replace("+", "", regex=False).str.replace(r"\d+", "", regex=True)
    holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

    # Bridge Holidays
    holidays["description"] = holidays["description"].str.replace("Puente ", "", regex=False)
    holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

    # Work Day Holidays, that is meant to payback the Bridge
    work_day = holidays[holidays.type == "Work Day"]
    holidays = holidays[holidays.type != "Work Day"]

    # Split
    # Events are national
    events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis=1).rename({"description":"events"}, axis=1)

    holidays = holidays[holidays.type != "Event"].drop("type", axis=1)

    # Only proceed if there are rows in holidays
    if not holidays.empty:
        regional = holidays[holidays.locale == "Regional"].rename({"locale_name":"state", "description":"holiday_regional"}, axis=1).drop("locale", axis=1).drop_duplicates()
        national = holidays[holidays.locale == "National"].rename({"description":"holiday_national"}, axis=1).drop(["locale", "locale_name"], axis=1).drop_duplicates()
        local = holidays[holidays.locale == "Local"].rename({"description":"holiday_local", "locale_name":"city"}, axis=1).drop("locale", axis=1).drop_duplicates()
    else:
        # Create empty DataFrames with the correct columns
        regional = pd.DataFrame(columns=["date", "state", "holiday_regional"])
        national = pd.DataFrame(columns=["date", "holiday_national"])
        local = pd.DataFrame(columns=["date", "city", "holiday_local"])

    # Merge National Holidays
    df = df.merge(national, how="left", on="date")

    # Regional - ensure 'state' column exists in both DataFrames
    if 'state' in df.columns:
        df = df.merge(regional, how="left", on=["date", "state"])
    else:
        # Add empty holiday_regional column if state column doesn't exist
        df['holiday_regional'] = np.nan

    # Local - ensure 'city' column exists in both DataFrames
    if 'city' in df.columns:
        df = df.merge(local, how="left", on=["date", "city"])
    else:
        # Add empty holiday_local column if city column doesn't exist
        df['holiday_local'] = np.nan

    # Work Day
    if not work_day.empty:
        df = df.merge(work_day[["date", "type"]].rename({"type":"IsWorkDay"}, axis=1), how="left")
    else:
        df["IsWorkDay"] = np.nan

    # EVENTS
    # Ensure events column is string type before using str methods
    if 'events' in events.columns and not events.empty:
        events['events'] = events['events'].fillna('').astype(str)
        events["events"] = np.where(events.events.str.contains("futbol", na=False), "Futbol", events.events)

        # One-hot encode events
        events_enc, events_cat = one_hot_encoder(events, nan_as_category=False)

        # Special case for Mother's Day
        if 'events_Dia_de_la_Madre' in events_enc.columns and len(events_enc) > 239:
            mother_day_date = pd.to_datetime("2016-05-08")
            events_enc.loc[events_enc.date == mother_day_date, 'events_Dia_de_la_Madre'] = 1
            if 239 < len(events_enc):
                events_enc = events_enc.drop(239)

        df = df.merge(events_enc, how="left", on="date")
        if events_cat:  # Only try to fill if the list is not empty
            df[events_cat] = df[events_cat].fillna(0)

    # New features - safely handle possibly missing columns
    if 'holiday_national' in df.columns:
        df["holiday_national_binary"] = np.where(df.holiday_national.notna(), 1, 0).astype('int8')
    else:
        df["holiday_national_binary"] = 0

    if 'holiday_local' in df.columns:
        df["holiday_local_binary"] = np.where(df.holiday_local.notna(), 1, 0).astype('int8')
    else:
        df["holiday_local_binary"] = 0

    if 'holiday_regional' in df.columns:
        df["holiday_regional_binary"] = np.where(df.holiday_regional.notna(), 1, 0).astype('int8')
    else:
        df["holiday_regional_binary"] = 0

    # Additional holiday features
    if 'holiday_national' in df.columns:
        df["national_independence"] = np.where(
            df.holiday_national.isin([
                'Batalla de Pichincha', 'Independencia de Cuenca', 
                'Independencia de Guayaquil', 'Primer Grito de Independencia'
            ]), 
            1, 0
        ).astype('int8')
    else:
        df["national_independence"] = 0

    # Process local holiday features if they exist
    if 'holiday_local' in df.columns:
        # First ensure column is string type and fill NaN values
        df['holiday_local'] = df['holiday_local'].fillna('').astype(str)

        # Now safely apply string methods
        df["local_cantonizacio"] = np.where(df.holiday_local.str.contains("Cantonizacio", na=False), 1, 0).astype('int8')
        df["local_fundacion"] = np.where(df.holiday_local.str.contains("Fundacion", na=False), 1, 0).astype('int8')
        df["local_independencia"] = np.where(df.holiday_local.str.contains("Independencia", na=False), 1, 0).astype('int8')
    else:
        df["local_cantonizacio"] = 0
        df["local_fundacion"] = 0
        df["local_independencia"] = 0

    # One-hot encode holiday columns if they exist
    holiday_cols = ["holiday_national", "holiday_regional", "holiday_local"]
    existing_holiday_cols = [col for col in holiday_cols if col in df.columns]

    if existing_holiday_cols:
        for col in existing_holiday_cols:
            # Ensure values are strings before one-hot encoding
            df[col] = df[col].fillna('').astype(str)

        # Now it's safe to one-hot encode
        holidays_enc, holidays_cat = one_hot_encoder(df[existing_holiday_cols], nan_as_category=False)
        df = pd.concat([df.drop(existing_holiday_cols, axis=1), holidays_enc], axis=1)

    # Convert holiday columns to int8
    he_cols = (
        df.columns[df.columns.str.startswith("events")].tolist() + 
        df.columns[df.columns.str.startswith("holiday")].tolist() + 
        df.columns[df.columns.str.startswith("national")].tolist() + 
        df.columns[df.columns.str.startswith("local")].tolist()
    )

    # Only convert columns that exist and contain numeric data
    existing_he_cols = [col for col in he_cols if col in df.columns]
    if existing_he_cols:
        for col in existing_he_cols:
            try:
                df[col] = df[col].fillna(0).astype("int8")
            except (ValueError, TypeError):
                # If conversion fails, leave as is
                pass

    return df, work_day


# --- Usage example ---
# Load data
# Ensure date/types
# ... previous merges ...
# Holiday processing
d, work_day = create_holiday_features(d, holidays)


# In[182]:


d.head()


# **Let's apply an AB test to Events and Holidays features. Are they statistically significant? Also it can be a good way for first feature selection.**
# 
# - *H0: The sales are equal* **(M1 = M2)**
# - *H1: The sales are not equal* **(M1 != M2)**

# In[183]:


def AB_Test(dataframe, group, target):

    # Packages
    from scipy.stats import shapiro
    import scipy.stats as stats

    # Split A/B
    groupA = dataframe[dataframe[group] == 1][target]
    groupB = dataframe[dataframe[group] == 0][target]

    # Assumption: Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05
    # H0: Distribution is Normal! - False
    # H1: Distribution is not Normal! - True

    if (ntA == False) & (ntB == False): # "H0: Normal Distribution"
        # Parametric Test
        # Assumption: Homogeneity of variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0: Homogeneity: False
        # H1: Heterogeneous: True

        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            # Heterogeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1] 
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True

    # Result
    temp = pd.DataFrame({
        "AB Hypothesis":[ttest < 0.05], 
        "p-value":[ttest]
    })
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!", "A/B groups are not similar!")
    temp["Feature"] = group
    temp["GroupA_mean"] = groupA.mean()
    temp["GroupB_mean"] = groupB.mean()
    temp["GroupA_median"] = groupA.median()
    temp["GroupB_median"] = groupB.median()

    # Columns
    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Feature","Test Type", "Homogeneity","AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
    else:
        temp = temp[["Feature","Test Type","AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]

    # Print Hypothesis
    # print("# A/B Testing Hypothesis")
    # print("H0: A == B")
    # print("H1: A != B", "\n")

    return temp

# Apply A/B Testing
he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist()+ d.columns[d.columns.str.startswith("local")].tolist()
ab = []
for i in he_cols:
    ab.append(AB_Test(dataframe=d[d.sales.notnull()], group = i, target = "sales"))
ab = pd.concat(ab)
ab


# In[184]:


d.groupby(["family","events_Futbol"]).sales.mean()[:60]


# # 7. Time Related Features
# 
# How many features can you create from only date column? I'm sharing an example of time related features. You can expand the features with your imagination or your needs. 

# In[185]:


def create_date_features(df):
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = df.date.dt.isocalendar().week.astype("int8")  # <-- FIXED HERE
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df

d = create_date_features(d)




# Workday column
d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary==1) | (d.holiday_regional_binary==1) | (d['day_of_week'].isin([6,7])), 0, 1)
d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
d.drop("IsWorkDay", axis = 1, inplace = True)

# Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. 
# Supermarket sales could be affected by this.
d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

d.head(15)


# # 8. Did Earhquake affect the store sales?
# 
# A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
# 
# Comparing average sales by year, month and product family will be one of the best ways to be able to understand how earthquake had affected the store sales.
# 
# We can use the data of March, April, May and June and there may be increasing or decrasing sales for some product families.
# 
# Lastly, we extracted a column for earthquake from Holidays and Events data. **"events_Terremoto_Manabi"** column will help to fit a better model.

# In[186]:


d[(d.month.isin([4,5]))].groupby(["year"]).sales.mean()


# ### March

# In[187]:


pd.pivot_table(d[(d.month.isin([3]))], index="year", columns="family", values="sales", aggfunc="mean")


# ### April - May

# In[188]:


pd.pivot_table(d[(d.month.isin([4,5]))], index="year", columns="family", values="sales", aggfunc="mean")


# ### June

# In[189]:


pd.pivot_table(d[(d.month.isin([6]))], index="year", columns="family", values="sales", aggfunc="mean")


# In[190]:


d.head()


# In[191]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split


# In[192]:


d.head()


# In[193]:


oil.isna().sum()


# In[194]:


oil.head()


# In[195]:


# Merge oil prices into d
d = d.merge(oil[['date', 'dcoilwtico_interpolated']], how='left', on='date')
# 1. Create binary feature
d['oil_above_70'] = (d['dcoilwtico_interpolated'] >= 70).astype('int8')

# 2. Drop the original
d = d.drop('dcoilwtico_interpolated', axis=1)
d.head()


# In[196]:


d['oil_above_70']


# In[197]:


def custom_log_callback(period=100):
    def callback(env):
        if env.iteration % period == 0:
            train_score = env.evaluation_result_list[0][2]  # train rmse
            valid_score = env.evaluation_result_list[1][2]  # valid rmse
            diff = valid_score - train_score
            print(f"[{env.iteration}] train's rmse: {train_score:.6f} | valid's rmse: {valid_score:.6f} | GAP (valid - train): {diff:.6f}")
    return callback


# In[198]:


# 🛠 Import libraries
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV



# In[199]:


# 📌 2. Separate train/test
train = d[d['sales'].notnull()]
test = d[d['sales'].isnull()]

# 📋 3. Feature selection
drop_cols = ['id', 'date', 'sales']
features = [col for col in train.columns if col not in drop_cols]

# 🎯 4. Target variable (log1p-transform)
y_train_full = np.log1p(train['sales'])

X_train_full = train[features]
X_test = test[features]

# ✂️ 5. Split into training/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# 📦 6. LightGBM Dataset
categorical_features = ['city', 'state', 'type', 'family', 'cluster', 'oil_above_70']

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=categorical_features)


print("🧩 Model will train on the following features:")
print(features)
print(len(features))

print("\n🔍 Sample of X_train data:")
print(X_train[features].head())

print("\n🎯 Sample of y_train data (log1p-transformed sales):")
print(y_train.head())



# In[200]:


# # ⚙️ 7. Updated LightGBM Parameters
# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.01,        # ⬇️ Lower for smoother learning
#     'num_leaves': 128,
#     'min_data_in_leaf': 30,        # ⬆️ Slightly increase
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'lambda_l1': 1,
#     'lambda_l2': 1,
#     'seed': 42,
#     'verbosity': -1
# }

# # 🏋️‍♂️ 8. Train the Model
# model = lgb.train(
#     params,
#     train_data,
#     num_boost_round=10000,           # Same max limit
#     valid_sets=[train_data, valid_data],
#     valid_names=['train', 'valid'],
#     callbacks=[
#         lgb.early_stopping(stopping_rounds=100),    # Early stopping if no improvement
#         custom_log_callback(period=100)    # 👈 use our custom logger
#     ]
# )

# # 🧠 9. Predict on Test
# test_preds = model.predict(X_test, num_iteration=model.best_iteration)

# # 🔙 10. Reverse log1p
# test_preds = np.expm1(test_preds)


# print("✅ Finished!")



# In[201]:


print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")

print(X_train.dtypes.value_counts())


# In[202]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 📊 1. Predict on Training and Validation sets
train_preds = model.predict(X_train, num_iteration=model.best_iteration)
valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)

# 📈 2. Reverse log1p transform
train_preds = np.expm1(train_preds)
train_labels = np.expm1(y_train)

valid_preds = np.expm1(valid_preds)
valid_labels = np.expm1(y_valid)

# 📏 3. Compute Metrics
# --- Train Metrics ---
train_rmse = np.sqrt(mean_squared_error(train_labels, train_preds))
train_mae = mean_absolute_error(train_labels, train_preds)
train_r2 = r2_score(train_labels, train_preds)

# --- Validation Metrics ---
valid_rmse = np.sqrt(mean_squared_error(valid_labels, valid_preds))
valid_mae = mean_absolute_error(valid_labels, valid_preds)
valid_r2 = r2_score(valid_labels, valid_preds)

# 🖨️ 4. Print
print("🏋️‍♂️ Train Metrics:")
print(f"   RMSE: {train_rmse:.5f}")
print(f"   MAE:  {train_mae:.5f}")
print(f"   R²:   {train_r2:.5f}")

print("\n🎯 Validation Metrics:")
print(f"   RMSE: {valid_rmse:.5f}")
print(f"   MAE:  {valid_mae:.5f}")
print(f"   R²:   {valid_r2:.5f}")


# Using Normalized RMSE = RMSE / (max value – min value)
# 

# In[ ]:


# 🔥 4. Normalize RMSE
sales_max = train_labels.max()  # could also use full sales from all train, if you prefer
sales_min = train_labels.min()
range_sales = sales_max - sales_min


train_rmse_normalized = train_rmse / range_sales
valid_rmse_normalized = valid_rmse / range_sales

# 🖨️ 5. Print
print(f"   Normalized RMSE Train: {train_rmse_normalized:.5f}")

print(f"   Normalized RMSE Validation: {valid_rmse_normalized:.5f}")


# In[ ]:


# 📋 Reverse log1p on train sales
real_sales = np.expm1(y_train_full)

# 📈 Sales range and statistics
print(f"Sales Min: {real_sales.min():.2f}")
print(f"Sales Max: {real_sales.max():.2f}")
print(f"Sales Mean: {real_sales.mean():.2f}")
print(f"Sales Median: {np.median(real_sales):.2f}")
print(f"Sales Std: {real_sales.std():.2f}")


# In[ ]:


import os
print(os.getcwd())


# In[ ]:


# Save model to a file
# model.save_model('lgb_model.txt')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# 📊 Plot Sales Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['sales'], bins=100, kde=True, color='blue')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[ ]:


# 📊 Plot Log1p(Sales) Distribution
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train['sales']), bins=100, kde=True, color='green')
plt.title('Log1p(Sales) Distribution')
plt.xlabel('log(1 + Sales)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

