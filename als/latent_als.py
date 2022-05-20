from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col, explode
import getpass
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import  RankingMetrics

from time import time 
from pyspark import SparkContext
from pyspark.sql.functions  import collect_list
import csv

def main(spark,netID):
    '''
    This file is used to calculate the latent features of the model. ALS from pyspark.ml.recommendation does not support the user Features() and productFeatures() function and hence this file is only used to get Latent features.
    '''
    train=spark.read.parquet('hdfs:/user/kp2670/sm-train').rdd
#     test=spark.read.parquet('hdfs:/user/kp2670/sm-test').rdd
#     validate=spark.read.parquet('hdfs:/user/kp2670/sm-val').rdd

    
    train = spark.createDataFrame(train)
#     train.show()
    listt = ['timestamp']
    train = train.drop(*listt)

    sc = SparkContext.getOrCreate()

    rank = 20
    reg=0.1
    numIterations = 5
    model = ALS.trainImplicit(train, rank, numIterations,alpha=reg)
    userFeat=model.userFeatures()
    prodFeat=model.productFeatures()
    userFeat = spark.createDataFrame(userFeat)
    prodFeat = spark.createDataFrame(prodFeat)

#     We get latent features as list and we used below code to explode the list and break it into columns.
    dfu = userFeat.select([F.col("_2")[i] for i in range(rank)])
    dfp = prodFeat.select([F.col("_2")[i] for i in range(rank)])
    dfu.write.csv('user_latent_features.csv')
    dfp.write.csv('product_latent_features.csv')
#     df2.show()

    
if __name__ == "__main__":

    # Create the spark session object
    spark=SparkSession.builder.appName('als').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark,netID)