from pyspark import SparkConf
from pyspark.sql import SparkSession
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
BUCKET = "dmacademy-course-assets" 
KEY1 = "vlerick/pre_release.csv"
KEY2 = "vlerick/after_release.csv"

if len(os.environ.get("AWS_SECRET_ACCESS_KEY")) < 1:

    config = {"spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
          "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.f3.s3a.InstanceProfileCredentialsProvider"
    }
else:
    config = {"spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1"
    }
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()
print("here!!!!!!!!!!!!!!!!!!!!!!!!")
df1 = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header = True)
print("AND here!!!!!!!!!!!!!!!!!!!!!!!!")
df2 = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header = True)
pre_df = df1.toPandas()
after_df = df2.toPandas()
print("Still working here")
merged_df = pd.merge(pre_df, after_df, how="inner", on=["movie_title"])
merged_df = merged_df.drop(['num_critic_for_reviews', 'gross', 'movie_title', \
   'num_voted_users', 'num_user_for_reviews', 'movie_facebook_likes'], axis = 1)
numeric_features=merged_df._get_numeric_data().columns.values.tolist()
print("Still working here2")
merged_df.drop_duplicates(inplace=True)
merged_df = merged_df.drop(['director_name', 'actor_2_name', 'actor_1_name', \
   'actor_3_name'], axis = 1)
merged_df = merged_df.drop(['actor_3_facebook_likes', 'actor_2_facebook_likes',\
   'actor_1_facebook_likes'], axis = 1)
merged_df.loc[(merged_df['country'] == "USA") & (merged_df['language'].isnull() == True), \
   'language'] = "English"
print("Still working here3")
merged_df = merged_df.replace(np.nan, 'Unrated')
merged_df = merged_df.replace('GP', 'PG')
merged_df = merged_df.replace('Not Rated', 'Unrated')
merged_df = merged_df.replace('Approved', 'Unrated')
merged_df = merged_df.replace('Passed', 'Unrated')
merged_df = merged_df.replace('X', 'IAC')
merged_df = merged_df.replace('R', 'IAC')
merged_df = merged_df.replace('NC-17', 'IAC')
print("Still working here4")
language_list = merged_df['language'].unique().tolist()
language_list.remove('English')
for language in language_list:
  merged_df = merged_df.replace(language, 'Others')
countries_list = merged_df['country'].unique().tolist()
countries_list.remove('USA')
for country in countries_list:
  merged_df = merged_df.replace(country, 'Others')
print("Still working here5")
genres_dummified = merged_df['genres'].str.get_dummies(sep = '|') 
combined_frames = [merged_df, genres_dummified]
combined_result = pd.concat(combined_frames, axis = 1)
print("Still working here6")
combined_result = combined_result.drop('genres', axis = 1)
rating_dummified = merged_df['content_rating'].str.get_dummies()
combined_frames = [combined_result, rating_dummified]
combined_result2 = pd.concat(combined_frames, axis = 1)
combined_result2 = combined_result2.drop('content_rating', axis = 1)
print("Still working here7")
le = LabelEncoder()
combined_result2['language'] = le.fit_transform(combined_result2['language'])
combined_result2['country'] = le.fit_transform(combined_result2['country'])
x = combined_result2.drop('imdb_score', axis = 1)
y = combined_result2['imdb_score']
x['duration'] = x['duration'].astype('float')
print("Still working here71")
x['director_facebook_likes'] = x['director_facebook_likes'].astype('float')
print("Still working here72")
x['cast_total_facebook_likes'] = x['cast_total_facebook_likes'].astype('float')
print("Still working here73")
x['budget'] = x['budget'].astype('float')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("Still working here8")
from xgboost import XGBRegressor
xgboost=XGBRegressor(n_estimators=50)
print("Still working here9")
xgboost.fit(pd.DataFrame(X_train)._get_numeric_data(), pd.DataFrame(y_train)._get_numeric_data())
print("Still working here10")
predictions=xgboost.predict(X_test)
print("Still working here11")
predictions = pd.DataFrame({"y_pred": predictions},index=X_test.index)
val_pred = pd.concat([pd.DataFrame(y_test),pd.DataFrame(predictions), pd.DataFrame(X_test)],axis=1)
print("Still working here12")
predictions_spark = spark.createDataFrame(val_pred)
x = predictions_spark.write.json(f"s3a://{BUCKET}/vlerick/aly_bayoumi/", mode="overwrite")




