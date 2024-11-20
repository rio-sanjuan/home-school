AWS Deequ is an open source library built on top of [[Apache Spark]] that lets us define unit tests for data and measure data quality in large datasets. Using Deequ unit tests, we can find anomalies and errors early, before the data gets used in model training. Deequ is designed to work with very large datasets (billions of rows). The open source library supports tabular data, i.e., CSV files, database tables, logs, or flattened JSON files. Anything we can fit in a Spark data frame, we can validate with Deequ.

We can leverage [[Amazon SageMaker#Processing Jobs]] support for Apache Spark to run our Deequ unit tests at scale. 

>In this setup, we don't need to provision any Apache Spark cluster ourselves, as SageMaker Processing handles the heavy lifting for us.

We can think of this approach as "serverless" Apache Spark.