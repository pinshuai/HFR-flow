#!/usr/bin/env python
# coding: utf-8

# # retrive **well data** from HEIS. 
# The common database names are listed here:
# 
# |Table name| Description|
# |----------|------------|
# |GW_SAMPLE_RESULTS_MV|HEIS--ALL groundwater chemistry data|
# |ALL_SAMPLE_RESULTS_MV|ALL chemistry data|
# |well_construction_mv|HEIS--well construction data, includes depth, casing, and other fields|
# |HYDRAULIC_HEAD_MV|HEIS--manual water level data|
# |WELL_ATTRIBUTES_MV|Well attribute data (elevation, depth, screen length)|
# |tblWells|HEIS--AWLN well information|
# |tblProcessedData|HEIS--AWLN raw data (hourly)|
# |vAwln|a view of Awln data that includes Hanford well names|
# 
# note: table name above begin with "v" indiates it is a view that Patrick created and it is not a token from HEIS, rather , it is a query based on the HEIS database.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy  as np
import pandas as pd
##, matplotlib, scipy, sqlalchemy as sa, pyodbc, psycopg2
import pyodbc as p
import sqlalchemy as sa
import matplotlib.pyplot as plt
import math


# In[2]:


out_dir = "/Users/shua784/Dropbox/PNNL/People/From_Patrick/SQL/"


# ## connect to HIES copy using SQLAlchemy 
# 
# common query method includes: `pd.read_sql_query`, `pd.read_sql_table`, `pd.read_sql`

# First, create engine using a typical database URL: `dialect + driver://username:password@host:port/database`. In our case, `mssql`is dialect (database name) and `pyodbc` is driver (the name of DBAPI). 

# In[2]:


# query different database

engine_ATLAS = sa.create_engine('mssql+pyodbc://Atlas_user:cY9@Mite@sqlsrvdev17:915/ATLAS?driver=ODBC Driver 13 for SQL Server')

engine_SOCRATES = sa.create_engine('mssql+pyodbc://socrates_user:RizeY&g9+Wist@sqlsrvdev17:915/SOCRATES?driver=ODBC Driver 13 for SQL Server')


# ### list all tables in the database

# In[3]:


select = engine_ATLAS.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME ASC")

tables = select.fetchall()
for row in tables:
    print(row)


# In[5]:


select = engine_SOCRATES.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME ASC")

tables = select.fetchall()
for row in tables:
    print(row)


# ### query a table and return as dataframe and save to csv file

# In[65]:


itable = 'awln'


# In[66]:


df = pd.read_sql_table(itable, engine_SOCRATES)
list(df)


# save dataframe to csv

# In[67]:


fname = out_dir + itable + ".csv"
df.to_csv(fname, index=False)


# ### show statistcs of sub table

# In[4]:


df= pd.read_sql_table('HYDRAULIC_HEAD_MV', engine_SOCRATES, schema='pflow', parse_dates='HYD_DATE_TIME_PST')

# print(df.head(20))
# list all columns within the table
list(df)


# In[20]:


myTable = df[['WELL_ID', 'WELL_NAME','HYD_DATE_TIME_PST','HYD_HEAD_METERS_NAVD88']]
myTable.columns = ['WELL_ID', 'WELL_NAME', 'DateTime', 'WL']


# In[21]:


print ('Here are the average min, mean, and max readings for each well')
avg_head = myTable.groupby(['WELL_ID']).agg(['min', 'mean', 'max'])
print (avg_head.head(5))


# ### plot subset of a table

# In[24]:


print ('here are readings and a chart for a specific well')

# plot selected well
well =myTable[myTable['WELL_ID'] == 'A4550'].copy()
# print (well.head(5))

well['DateTime'] = well['DateTime'].apply(pd.to_datetime)

fig, ax = plt.subplots(1, 1)

plt.plot(well.DateTime, well.WL, '-ok')


# ## connect to HEIS using pyodbc

# In[25]:


connection = p.connect('DRIVER={ODBC Driver 13 for SQL Server};'
'SERVER=SQLSRVDEV17.PNL.GOV,915;'
'DATABASE=SOCRATES;'
'UID=socrates_user;'
'PWD=RizeY&g9+Wist')


# In[26]:


#create cursor instance
cursor = connection.cursor()


# ### list all tables

# In[28]:


list(cursor.tables())


# In[41]:


#%%################### print header in a table###################################
for row in cursor.columns(table='HYDRAULIC_HEAD_MV'):
   print (row.column_name)


# ### write table to csv file

# In[42]:


#---------Select non-duplicate Data from column and Rank by ID---------------#
select = cursor.execute(
    "SELECT * FROM dbo.HYDRAULIC_HEAD_MV ORDER BY dbo.HYDRAULIC_HEAD_MV.WELL_ID")

rows = select.fetchall()
for row in rows:
    print(row)

with open("HYDRAULIC_HEAD_MV_WellID.csv", "w", newline='') as csv_file:  # Python 3 version
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(rows)


# In[ ]:


# print header
for row in cursor.columns(table='mvAwln'):
    print(row.column_name)
# print table
select = cursor.execute("SELECT TOP (10) dbo.mvAwln.* FROM dbo.mvAwln")
rows = select.fetchall()
for row in rows:
    print(row)


# ### export as dataframe and plot water level

# In[ ]:


#%%##########################Select Conditional Data from Table##################
select=cursor.execute("SELECT dbo.mvAwln.WellName, dbo.mvAwln.WellNumber, dbo.mvAwln.procWaterElevation, dbo.mvAwln.procDate FROM dbo.mvAwln WHERE dbo.mvAwln.procDate >= '2009-01-01'")
rows = select.fetchall()

####################sort col based on datetime#################################
df = pd.DataFrame( [[ij for ij in i] for i in rows] )
df.rename(columns={0: 'WellName', 1: 'WellID', 2: 'WL', 3: 'DateTime'}, inplace=True);
df['DateTime'] =pd.to_datetime(df.DateTime)
df = df.sort_values(['DateTime'], ascending=[1]);
df = df.reset_index(drop=True)
######################creat plot###############################################


# In[44]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.plot(df.DateTime, df.WL)
# plt.show()


# In[ ]:


with open("mvAwln_since_2009.csv", "w", newline='') as csv_file:  
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in cursor.description]) # write headers
    csv_writer.writerows(rows)


# In[ ]:


cursor.close()
connection.close() #llose current connection

