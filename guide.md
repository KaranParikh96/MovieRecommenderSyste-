# Basic recommender system

## popularity baseline model

* Run `popularity_based_movies_recommendation.py` to get top 100 movies accross all users. 
 
## Train test split.

* To split the data into `train` , `validation` and `test` set run the `train_test_split.py`. The data will be stored in the HDFS and as a parquet files.

## ALS

* First of all run `hypertune_als.py` to perform hyper parameter tuning. YOu will get the best parameter printed in the end. 
* Using the best paramters run `test_als.py` and get top 100 movies per user on the test and validation dataset. It will also output the ranking metrics on both.`test_output-sm` containes test set predictions and `val-output-sm` contains validation set predictions. I had to remove those files due to the amount of space they took in the git.
* To get the latent features of the model, run `latent_als.py`. Latent features are then used in the visualization.

## Extension 1 (LighFM)

* First of all run `lightfm_sampler.ipynb` to sample the data. 
* After sampling run `lightfm_repartition.ipynb` to split the data
* Now run `createcsv.ipynb` to combine all the partition data into 1 csv
* Lastly, run `lightfm_singlemachine.ipynb` to get comparison of single machine model with ALS. 