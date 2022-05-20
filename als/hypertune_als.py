from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
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

from pyspark.mllib.evaluation import  RankingMetrics

from time import time 
from pyspark import SparkContext
from pyspark.sql.functions  import collect_list
import csv


def main(spark,netID):
    '''
    This file is used to hyper tune the model on three parameters : Rank , max_Iter and regularization. It perform training n different permutation of parameter and validation set is used to know the results of the training. We choose the best paramers using the ranking metrics we get on validation set.
    
    '''


    train=spark.read.parquet('hdfs:/user/kp2670/sm-train').rdd
    test=spark.read.parquet('hdfs:/user/kp2670/sm-test').rdd
    validate=spark.read.parquet('hdfs:/user/kp2670/sm-val').rdd
    
    loss=100

    train = spark.createDataFrame(train)
    test = spark.createDataFrame(test)
    validate = spark.createDataFrame(validate)
    
    train=train.na.drop()
    test=test.na.drop()
    validate=validate.na.drop()
    validate.createOrReplaceTempView('validate')
    models = {}
    precision_at_k_scores = {}
    maps ={}
    NDCGs = {}
    times = {}
    sc = SparkContext.getOrCreate()
#     param_reg=[0.001,0.01,0.1]
#     param_rank=[20,40,60,80]
#     param_maxIter=[10,15,20]
    param_reg=[0.1]
    param_rank=[20]
    param_maxIter=[5,10]
    for r in param_rank:
        for Iter in param_maxIter:
            for reg in param_reg:
                st = time()
                als = ALS(rank=r, maxIter=Iter , regParam=reg , userCol="userId", itemCol="movieId",
                          ratingCol="rating", nonnegative = True, implicitPrefs = False,
                          coldStartStrategy="drop")
                alsmodel = als.fit(train)
                # print(best_model.getParam('regParam'))
                models[(r,Iter,reg)] = alsmodel
                # Evaluate the model by computing the RMSE on the test data
                predictions = alsmodel.transform(test)
                # predictions = predictions.na.drop()
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                                predictionCol="prediction")
                rmse = evaluator.evaluate(predictions)
                
                if rmse<loss:
                    loss=rmse
                    best_params=(r,Iter,reg)
                    
                print("Root-mean-square error = " + str(rmse))
                
                preds = alsmodel.recommendForAllUsers(100)
                preds.createOrReplaceTempView('preds')
                val = spark.sql('SELECT userId, movieId FROM validate SORT BY rating DESC')
                val = val.groupBy('userId').agg(collect_list('movieId').alias('movieId_val'))
                val.createOrReplaceTempView('val')
                predAndTruth = spark.sql('SELECT preds.recommendations, val.movieId_val FROM val join preds on preds.userId = val.userId')
                predAndTruth = predAndTruth.collect()
                final_predAndTruth = []
                for item in predAndTruth:
                    truth = item[1]
                    pred = [i.movieId for i in item[0]]
                    final_predAndTruth += [(pred,truth)]


                final_predAndTruth =  sc.parallelize(final_predAndTruth)

                ranking_obj = RankingMetrics(final_predAndTruth)
                precision_at_k_scores[(r,Iter,reg)] = ranking_obj.precisionAt(100)
                maps[(r,Iter,reg)] = ranking_obj.meanAveragePrecision
                NDCGs[(r,Iter,reg)] = ranking_obj.ndcgAt(100)
                times[(r,Iter,reg)] = round(time() - st,5)

                print('Model with maxIter = {}, reg = {}, rank = {} complete'.format(Iter,reg,r))
                
                print('precision_at_k_scores',ranking_obj.precisionAt(100))
                print('MAP',ranking_obj.meanAveragePrecision)
                print('NDCGs',ranking_obj.ndcgAt(100))
                print('times',round(time() - st,5))
    print('best parameters',best_params)
                

if __name__ == "__main__":

    # Create the spark session object
    spark=SparkSession.builder.appName('als').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark,netID)