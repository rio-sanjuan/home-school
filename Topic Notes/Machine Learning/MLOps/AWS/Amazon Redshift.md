Amazon Redshift is a fully managed cloud data warehouse services that allows us to run complex analytic queries against petabytes of structured data. Our queries are distributed and parallelized across multiple nodes. In contrast to relational databases that are optimized to store data in rows and mostly serve transactional applications, Redshift implements columnar data storage, which is optimized for analytical applications where we are mostly interested in the summary statistics on those columns.

# Spectrum

Allows us to directly execute SQL queries from Redshift against exabytes of unstructured data in our [[Amazon S3]] data lake without the need to physically move the data. Amazon Redshift Spectrum automatically scales the compute resources needed based on how much data is being received, so queries against S3 run fast, regardless of the size of our data.