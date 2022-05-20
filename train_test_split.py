import sys
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, last
from pyspark.sql.functions import countDistinct
import os

def main(spark, netID):
    '''
    This file is used to split the data into train , validation and test sets.
    '''
    # read data
    interaction = ratings = spark.read.csv(f'hdfs:/user/{netID}/ml-latest/ratings.csv', schema='userId INT, movieId INT, rating DOUBLE, timestamp INT')
#     interaction = ratings = spark.read.csv(f'hdfs:/user/{netID}/ml-latest-small/ratings.csv', schema='userId INT, movieId INT, rating DOUBLE, timestamp INT')

    # subsampling
    partial=1.0
    interaction = interaction.sample(False, partial, seed = 2020)
    interaction.createOrReplaceTempView('interaction')

    # delete rating = 0
    rating_interaction = spark.sql('SELECT * FROM interaction WHERE rating != 0')
    rating_interaction.createOrReplaceTempView('rating_interaction')

    # select users with 10 or more valid interactions
    user = spark.sql('SELECT DISTINCT(userId) FROM rating_interaction GROUP BY userId HAVING count(*) >= 10')
    user.createOrReplaceTempView('user')

    # 60% training users, 20% val users, 20% test users
    train_user, val_user, test_user = user.randomSplit([0.6, 0.2, 0.2], seed = 2020)
    train_user.createOrReplaceTempView('train_user')
    val_user.createOrReplaceTempView('val_user')
    test_user.createOrReplaceTempView('test_user')

    training_from_train = spark.sql('SELECT * FROM rating_interaction WHERE userId IN (SELECT userId FROM train_user)')
    training_from_train.createOrReplaceTempView('training_from_train')
    all_val_interaction = spark.sql('SELECT * FROM rating_interaction WHERE userId IN (SELECT userId FROM val_user)')
    all_test_interaction = spark.sql('SELECT * FROM rating_interaction WHERE userId IN (SELECT userId FROM test_user)')

    # select half interaction from val and test interactions to training
    # val
    all_val_interaction_rdd = all_val_interaction.rdd.zipWithIndex()
    all_val_interaction_rdd_final = all_val_interaction_rdd.toDF()
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('userId', all_val_interaction_rdd_final['_1'].getItem("userId"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('movieId', all_val_interaction_rdd_final['_1'].getItem("movieId"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('rating', all_val_interaction_rdd_final['_1'].getItem("rating"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('timestamp', all_val_interaction_rdd_final['_1'].getItem("timestamp"))

    temp_val_interaction = all_val_interaction_rdd_final.select('_2', 'userId', 'movieId', 'rating', 'timestamp')
    temp_val_interaction.createOrReplaceTempView('temp_val_interaction')
    temp_even_val_interaction = spark.sql('SELECT * FROM temp_val_interaction WHERE _2 %2 =0')
    temp_odd_val_interaction = spark.sql('SELECT * FROM temp_val_interaction WHERE _2 %2 =1')

    # test
    all_test_interaction_rdd = all_test_interaction.rdd.zipWithIndex()
    all_test_interaction_rdd_final = all_test_interaction_rdd.toDF()
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('userId', all_test_interaction_rdd_final['_1'].getItem("userId"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('movieId', all_test_interaction_rdd_final['_1'].getItem("movieId"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('rating', all_test_interaction_rdd_final['_1'].getItem("rating"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('timestamp', all_test_interaction_rdd_final['_1'].getItem("timestamp"))

    temp_test_interaction = all_test_interaction_rdd_final.select('_2', 'userId', 'movieId', 'rating', 'timestamp')
    temp_test_interaction.createOrReplaceTempView('temp_test_interaction')
    temp_even_test_interaction = spark.sql('SELECT * FROM temp_test_interaction WHERE _2 %2 =0')
    temp_odd_test_interaction = spark.sql('SELECT * FROM temp_test_interaction WHERE _2 %2 =1')


    temp_even_test_interaction =  temp_even_test_interaction.drop('_2')
    temp_odd_test_interaction = temp_odd_test_interaction.drop('_2')
    temp_odd_val_interaction =  temp_odd_val_interaction.drop('_2')
    temp_even_val_interaction =  temp_even_val_interaction.drop('_2')
    temp_even_val_interaction.createOrReplaceTempView('temp_even_val_interaction')
    temp_odd_val_interaction.createOrReplaceTempView('temp_odd_val_interaction')
    temp_odd_test_interaction.createOrReplaceTempView('temp_odd_test_interaction')
    temp_even_test_interaction.createOrReplaceTempView('temp_even_test_interaction')


    training = spark.sql('SELECT * FROM training_from_train UNION ALL SELECT * FROM temp_even_val_interaction UNION ALL SELECT * FROM temp_even_test_interaction')
    validation = temp_odd_val_interaction
    testing = temp_odd_test_interaction
    
        
    training.write.option("header","true").mode("overwrite").parquet("lr-train")
#     training.write.option("header","true").mode("overwrite").parquet("sm-train")
    
    validation.write.option("header","true").mode("overwrite").parquet("lr-val")
#     validation.write.option("header","true").mode("overwrite").parquet("sm-val")
    
    testing.write.option("header","true").mode("overwrite").parquet("lr-test")
#     testing.write.option("header","true").mode("overwrite").parquet("sm-test")
    
#     training.show()
#     validation.show()
#     testing.show()
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test_partition').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)