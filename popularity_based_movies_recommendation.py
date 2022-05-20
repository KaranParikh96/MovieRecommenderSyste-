import sys
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, last
from pyspark.sql.functions import countDistinct
import os
import numpy as np
import pandas
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank, row_number, asc, desc, col, monotonically_increasing_id, dense_rank
# import Recommender

import pandas as pd

'''
This file is used to calculate the popularity baseline model.
'''
class Popularity_Recommender():

    # Initialize all the variables
    def __init__(self):
        # Tha training data which is been provided.
        self.train_data = None

        # The id of the user for which the recommendations is needed.
        self.user_id = None

        # The id of item e.g. Movies
        self.item_id = None

        # The final result which is going to be returned as a dataframe. 
        self.popularity_recommendataions = None

    # Create the recommendations.
    def create(self,train_data,user_id,item_id):

        # init The training data 
        self.train_data = train_data

        # The recommendation for userid
        self.user_id = user_id

        # The id of item e.g Movies
        self.item_id = item_id

        # The index is reset once the items are grouped by item id and aggregated with the number of users.

        train_data = train_data.withColumn("userId2", train_data["userId"])
        train_data_grouped = train_data.select("*").groupBy([self.item_id]).agg(F.count("userId2").alias("score"))
        
        
        
        # The training data is arranged in ascending order by item id and descending order by score.
        train_data_sort = train_data_grouped.sort(desc("score"), asc("movieId"))
        # Score is sorted in ascending order to create the new column Rank.
        
        train_data_sort = train_data_sort.limit(100)
        train_data_sort = train_data_sort.toPandas()
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending = 0, method = 'first')
        print(train_data_sort.head(100))
        print(train_data_sort.columns)
   


        # The first 100 items are saved into the popularity_recommendataions and it is returned. 
        self.popularity_recommendataions = train_data_sort


    # Method to user created recommendations
    def recommend(self, user_id):

        # Since the suggestions have been recorded into this column, populate the user recommendataion field with popularity recommendataions.
        user_recommendataion = self.popularity_recommendataions
        print(user_recommendataion.columns)

        # fetch the user_id
        newcol = 100*[user_id]
        newcol = np.array(newcol, dtype=int)

        user_recommendataion['userId'] = newcol

        # column setter
        cols = user_recommendataion.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendataion = user_recommendataion[cols]

        return user_recommendataion


def main(spark, netID):
    train_path = f'hdfs:/user/{netID}/sm-train'
    train_df = spark.read.option("header","true").option("recursiveFileLookup","true").parquet(train_path)
    train_df.printSchema()
    
    valid_path = f'hdfs:/user/{netID}/sm-val'
    valid_df = spark.read.option("header","true").option("recursiveFileLookup","true").parquet(valid_path)
    valid_df.printSchema()
    
    test_path = f'hdfs:/user/{netID}/sm-test'
    test_df = spark.read.option("header","true").option("recursiveFileLookup","true").parquet(test_path)
    test_df.printSchema()

    
#     pr = Recommender.Popularity_Recommender()
    pr = Popularity_Recommender()
    pr.create(train_df, 'userId', 'movieId')
    

    users = valid_df.select('userId').distinct().collect()
    result_valid = pr.recommend(users[5])
    
#     result_valid.to_csv("../result_valid_large.csv", header=True, index=True)
    print(pr.recommend(users[5]).head(100))
      
    
    users = test_df.select('userId').distinct().collect()
    result_test = pr.recommend(users[5])
    
#     result_test.to_csv("../result_test_large.csv", header=True, index=True)
    print(pr.recommend(users[1]).head(100))
    
    result_validsparkDF2 = spark.createDataFrame(result_valid)
    result_validsparkDF2.printSchema()
    result_validsparkDF2.show()
    for col in result_validsparkDF2.dtypes:
        print(col[0]+" , "+col[1])
    
    
    result_testsparkDF = spark.createDataFrame(result_test) 
    result_testsparkDF.printSchema()
    result_testsparkDF.show()
    
    for col in result_testsparkDF.dtypes:
        print(col[0]+" , "+col[1])
    
    result_validsparkDF2.coalesce(1).write\
      .option("header","true")\
      .option("sep",",")\
      .mode("overwrite")\
      .csv("result_valid_small.csv")
#     result_validsparkDF2.coalesce(1).write\
#       .option("header","true")\
#       .option("sep",",")\
#       .mode("overwrite")\
#       .csv("result_valid_large.csv")
    
    result_testsparkDF.coalesce(1).write\
      .option("header","true")\
      .option("sep",",")\
      .mode("overwrite")\
      .csv("result_test_small.csv")
#     result_testsparkDF.coalesce(1).write\
#       .option("header","true")\
#       .option("sep",",")\
#       .mode("overwrite")\
#       .csv("result_test_large.csv")
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)