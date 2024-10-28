
Extends the power of [[Pandas]] to AWS. AWS Data Wrangler connects pandas *DataFrames* with AWS services such as [[Amazon S3]], [[AWS Glue]], [[Amazon Athena]], and [[Amazon Redshift]].

AWS Data Wrangler provides optimized Python functions to perform common ETL tasks to load and unload data between data lakes, data warehouses, and databases.

```bash
pip install awswrangler
```

After installing, we can read our data directly from S3 into a pandas DataFrame as shown here:

```python
import awswrangler as wr

# Retrieve the data directly from Amazon S3
df = wr.s3.read_parquet("s3://<BUCKET>/<DATASET>/")
```

AWS Data Wrangler also comes with additional memory optimizations, such as reading data in chunks. This is particularly helpful if we need to query large datasets. With chunking enables, AWS Data Wrangler reads and returns every dataset file in the path as a separate pandas DataFrame. We can also set the chunk size to return the number of rows in a DataFrame equivalent to the numerical value we defined as chunk size.

See more in [[Data Science on AWS#Chapter 5 Explore the Dataset]].