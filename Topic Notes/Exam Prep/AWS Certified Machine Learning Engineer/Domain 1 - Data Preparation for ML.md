
# 1.1: Ingest and Store Data

The steps of the data engineering lifecycle are generation, storage, ingestion, transformation, and serving data.
### Step 1: Generation

The first step is generation, which is where the data originates from. So it's your source system such as a database, IoT device, and more. And for the generation stage, the data engineer or the machine learning engineer consumes the data from a source system, but might not control that source system.
### Step 2: Storage (Ingestion, Transformation, Serving)

The second step is Storage. For this exam, ensure you can choose an appropriate data store to store the data. Also ensure you understand different use cases and different AWS storage services to ingest your data from the source system into AWS, and ensure you understand how to extract that data and ingest and process that data into your machine learning algorithm to train your model.
#### Storage Abstractions
* Data lake
* Data lakehouse
* Data platform
* Cloud data warehouse
#### Storage systems
* HDFS
* RDBMS
* Object storage
* Streaming storage
* Cache- or memory-based storage
#### Raw ingredients
* SDD
* HDD
* RAM
* CPU
* Networking
* Compression
* Serialization

>Which AWS storage service is best for your use case? Which format is the data in? What are the access patterns? Is the data streaming? Does the data need to be merged from multiple sources?
### AWS storage and database services

#### Object
1. [[Amazon S3]]
#### Block
1. [[Amazon EBS]]
#### File
1. [[Amazon EFS]]
2. [[Amazon FSx]]
#### Backup
1. [[AWS Backup]]
#### Data Transfer and Migration
1. [[AWS Storage Gateway]]
2. [[AWS DataSync]]
3. [[AWS Transfer Family]]
4. [[AWS Snowball]]
5. [[AWS Snowcone]]
#### Other
1. [[Amazon RDS]]
2. [[Amazon DynamoDB]]
3. [[Amazon Redshift]]

For this exam, you need to understand the use cases and trade-offs for the AWS storage services. And for cost optimization, understand the different storage classes of those services and the lifecycle policies.
## Choosing an AWS storage service

Resources:
1. [AWS Summit ATL 2022 - Choosing the right AWS storage service for the job](https://www.youtube.com/watch?v=A14EbSrZeFM&t=16s)
2. [Choosing an AWS storage service (whitepaper)](https://docs.aws.amazon.com/decision-guides/latest/storage-on-aws-how-to-choose/choosing-aws-storage-service.html) 

AWS offers a broad portfolio of reliable, scalable, and secure storage services for storing, accessing, protecting, and analyzing your data. This makes it easier to match your storage methods with your needs, and provides storage options that are not easily achievable with on-premises infrastructure. When selecting a storage service, ensuring that it aligns with your access patterns will be critical to achieving the performance you want. You can select from block, file, and object storage services as well as cloud data migration options for your workload. Choosing the right storage service for your workload requires you to make a series of decisions based on your business needs.

## Use Cases

```
You need to store unprocessed data from IoT devices for a new machine learning pipeline. The storage solution should be a centralized and highly available repository.

What AWS storage service do you choose to store the unprocessed data?
```

[[Amazon EFS]] or [[Amazon S3]]?

Let's point out a few keywords in this question, which are "store unprocessed data," "pipeline," and "centralized," and "highly available repository." Do you store the unprocessed data in an Amazon Elastic File System, Amazon EFS, or do you store the data in Amazon S3 and create a data lake for the unprocessed data? You could use Amazon EFS to store the data, but Amazon S3 is a better choice because it has direct integration with other AWS services, such as AWS Glue, Amazon EMR, Amazon SageMaker and more, that you can use to build a machine learning pipeline. And remember that "pipeline" was a keyword. Amazon S3 has a use case for unstructured data, and again, integrates with other AWS services to build your pipeline. Also, another key phrase was unprocessed data. Amazon EFS supports file storage and a use case is scalable data storage.

```
You are designing a highly scalable data repository for your machine learning pipeline. You need immediate access to the processed data from your pipeline for 6 months. Your unprocessed data must be accessible within 12 hours and stored for 6 years.
```

What is your cost-effective storage solution?

One solution is to set an S3 lifecycle with lifecycle configuration rules that defines actions to store your objects in S3. For this solution, you can store your processed data in an Amazon S3 Standard storage class, create a lifecycle policy to move your processed data into Amazon S3 Glacier after 6 months. And for the unprocessed data, you can create another lifecycle policy to move that data into Amazon S3 Glacier Deep Archive.

Let's pause and look at two key phrases for this scenario question. "Immediate access" and "highly available." So for this question, Amazon S3 One-Zone storage class would not be a best-choice answer, because with both of the One-Zone storage class options, your data is only stored in one availability zone.

What if we added another requirement for the storage solution to support SQL querying capabilities? Well, with Amazon S3, you can use [[Amazon Athena]] for ad hoc SQL queries. You cannot run SQL queries in [[Amazon DynamoDB]] unless you use [[PartiQL]]. And for [[Amazon Redshift]], it does support SQL queries, but would not be the best-choice answer.

# 1.2: Transform Data and Perform Feature Engineering