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
    Takes in the train test and valudation set, trains the model on the best param found in the
    hyper parameter tuning and makes predictions. Predictions are then ised to calculate the ranking matrics.
    In the end we predict top 100 movies per user and save the results in csv.
    '''
    train=spark.read.parquet('hdfs:/user/kp2670/lr-train').rdd
    test=spark.read.parquet('hdfs:/user/kp2670/lr-test').rdd
    validate=spark.read.parquet('hdfs:/user/kp2670/lr-val').rdd


    train = spark.createDataFrame(train)
    test = spark.createDataFrame(test)
    validate = spark.createDataFrame(validate)

    train=train.na.drop()
    test=test.na.drop()
#     validate=validate.na.drop()
    test.createOrReplaceTempView('test')
    validate.createOrReplaceTempView('validate')

    sc = SparkContext.getOrCreate()
    st = time()
    Iter=10
    reg=0.1
    r=20
    als = ALS(rank=r, maxIter=Iter , regParam=reg , userCol="userId", itemCol="movieId",
                              ratingCol="rating", nonnegative = True, implicitPrefs = False,
                              coldStartStrategy="drop")
    alsmodel = als.fit(train)
    # print(best_model.getParam('regParam'))
#     models[(r,Iter,reg)] = alsmodel
    # Evaluate the model by computing the RMSE on the test data
    predictions = alsmodel.transform(test)
    # predictions = predictions.na.drop()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print('calculating ranking matrics on test data')
    print("Root-mean-square error = " + str(rmse))
    
    # evaluate on validation 
    preds = alsmodel.recommendForAllUsers(100)
    preds.createOrReplaceTempView('preds')
    val = spark.sql('SELECT userId, movieId FROM test SORT BY rating DESC')
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
    precision_at_k_scores = ranking_obj.precisionAt(100)
    maps = ranking_obj.meanAveragePrecision
    NDCGs = ranking_obj.ndcgAt(100)
    times = round(time() - st,5)

    print('precision_at_k_scores',precision_at_k_scores)
    print('MAP',maps)
    print('NDCGs',NDCGs)
    print('times',times)

    #to get top 100 movies ratings of all users
    recommendations  = alsmodel.recommendForAllUsers(100)
#     userRecs.filter(userRecs['userId']==11).show()
#     userRecs.show()
#     type(userRecs)
#     output=userRecs.createDataFrame(userRecs)
#     output.show()
    recommendations = recommendations\
        .withColumn("rec_exp", explode("recommendations"))\
        .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
#     recommendations.show()
    recommendations.write.csv('test-output-lr')
    
    ## for validation data
    print('calculating ranking matrics on validation data')
    
    predictions = alsmodel.transform(test)
    # predictions = predictions.na.drop()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    
    #to get top 100 movies ratings of all users
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
    precision_at_k_scores = ranking_obj.precisionAt(100)
    maps = ranking_obj.meanAveragePrecision
    NDCGs = ranking_obj.ndcgAt(100)
    times = round(time() - st,5)

    print('precision_at_k_scores',precision_at_k_scores)
    print('MAP',maps)
    print('NDCGs',NDCGs)
    print('times',times)

    recommendations  = alsmodel.recommendForAllUsers(100)
#     userRecs.filter(userRecs['userId']==11).show()
#     userRecs.show()
#     type(userRecs)
#     output=userRecs.createDataFrame(userRecs)
#     output.show()
    recommendations = recommendations\
        .withColumn("rec_exp", explode("recommendations"))\
        .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
#     recommendations.show()
    recommendations.write.csv('val-output-lr')
    
if __name__ == "__main__":

    # Create the spark session object
    spark=SparkSession.builder.appName('als').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark,netID)