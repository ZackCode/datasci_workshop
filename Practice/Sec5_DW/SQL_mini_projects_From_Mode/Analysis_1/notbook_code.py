# Python Notebook - Investigating a Drop in User Engagement

datasets

import pandas as pd
import matplotlib.pyplot as plt
# getting the data ready
df = pd.DataFrame(datasets['Problem'])
df['date_trunc'] = pd.to_datetime(df['date_trunc'])
df = df.set_index('date_trunc')
# plot
plt.style.use('ggplot')
plt.plot(df)
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('User Count')
plt.title('Weekly Active Users')
plt.show()

df_new = pd.DataFrame(datasets['New_User_Register'])
df_new['date_trunc'] = pd.to_datetime(df_new['date_trunc'])
df_new = df_new.set_index('date_trunc')
# plot
plt.style.use('ggplot')
plt.plot(df, label='weekly active users')
plt.plot(df_new, label='weekly signup users')
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('User Count')
plt.title('Weekly Active Users')
plt.legend(loc=4)
plt.show()

df_com = pd.DataFrame(datasets['Companies'])
df_com['date_trunc'] = pd.to_datetime(df_com['date_trunc'])
df_com = df_com.set_index(['date_trunc','company_id']).unstack()

# plot
plt.style.use('ggplot')
df_com.plot(kind='bar',figsize=(20,10))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('User Count')
plt.title('Weekly Active Users')
plt.show()

df_general = pd.DataFrame(datasets['General_Service'])
df_general['date_trunc'] = pd.to_datetime(df_general['date_trunc'])
df_general = df_general.set_index('date_trunc')
# plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.yscale('log')
plt.plot(df_general)
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('User Count/Precentage')
plt.title('General Service')
plt.legend(df_general.columns.values,loc=5)
plt.show()

df_email = pd.DataFrame(datasets['Email_Service'])
df_email['date_trunc'] = pd.to_datetime(df_email['date_trunc'])
df_email = df_email.set_index('date_trunc')
# plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.yscale('log')
plt.plot(df_email)
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Action Count')
plt.title('Email Service')
plt.legend(df_email.columns.values,loc=4)
plt.show()

### check further problem

df_de = pd.DataFrame(datasets['Deeper'])
df_de['date_trunc'] = pd.to_datetime(df_de['date_trunc'])
df_de = df_de.set_index('date_trunc')
# plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.plot(df_de)
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('User Count')
plt.title('Deeper Look')
plt.legend(df_de.columns.values,loc=4)
plt.show()

