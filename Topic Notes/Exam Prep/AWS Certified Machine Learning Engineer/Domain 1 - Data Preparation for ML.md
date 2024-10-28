
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

## Programming Concepts in AWS

* **Ingestion**: [[AWS Lambda]]
* **Processing**: [[Amazon EMR]], [[AWS Glue]], [[Apache Spark]], [[Amazon Kinesis]]
* **Transformations**: AWS Glue
* **Storage**: [[Amazon S3]], [[Amazon RDS]], [[Amazon Redshift]]
* **Orchestration**: [[AWS Step Functions]]
## AWS storage and database services

#### Object
1. [[Amazon S3]]

Stores data as objects, where each object consists of a file, metadata, and a unique identifier. Ideal for unstructured data like images, videos, backups, big data analytics, and static web content. Highly scalable with built-in data redundancy. Highly scalable with built-in data redundancy. Accessed through an API (HTTP/HTTPS), not directly mountable. Cost-effective, with features like versioning, cross-region replication, and fine-grained access control.
#### Block
1. [[Amazon EBS]]

Stores data in blocks, similar to a physical hard drive, allowing data to be split and distributed in blocks. Suitable for high-performance applications like databases, virtual machines, and applications needed low-latency, persistent storage. Scales based on volume, and is generally used with a single [[Amazon EC2]] instance. Mountable as a file system by an operating system. High-performance I/O operations, consistency, and low-latency access, making it ideal for applications requiring quick data retrieval.
#### File
1. [[Amazon EFS]]
2. [[Amazon FSx]]

Stores data in a hierarchical file system structure, similar to traditional file systems. Best for shared storage across multiple instances, enterprise applications, media processing, and content management. Automatically scalable, allowing multiple EC2 instances to access the same file system. It supports [[Network File System (NFS)]] (EFS) or [[Server Message Block (SMB)]] (FSx for Windows). Offers shared access to files with scalable performance, seamless integration with AWS services, and compatibility with applications requiring file-based storage.
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

### Structured Data
* [[Amazon RDS]]
* [[Amazon Aurora]]
* [[Amazon Redshift]]
* [[Amazon S3]]
* [[Amazon Athena]]
### Semi-Structured
* [[Amazon DynamoDB]]
* [[Amazon DocumentDB]]
* Amazon Athena
* Amazon S3
### Unstructured
* Amazon S3
* [[Amazon Rekognition]]
* [[Amazon Transcribe]]
* [[Amazon Comprehend]]
## Resources for Choosing an AWS Storage Service

1. [AWS Summit ATL 2022 - Choosing the right AWS storage service for the job](https://www.youtube.com/watch?v=A14EbSrZeFM&t=16s)
2. [Choosing an AWS storage service (whitepaper)](https://docs.aws.amazon.com/decision-guides/latest/storage-on-aws-how-to-choose/choosing-aws-storage-service.html) 

AWS offers a broad portfolio of reliable, scalable, and secure storage services for storing, accessing, protecting, and analyzing your data. This makes it easier to match your storage methods with your needs, and provides storage options that are not easily achievable with on-premises infrastructure. When selecting a storage service, ensuring that it aligns with your access patterns will be critical to achieving the performance you want. You can select from block, file, and object storage services as well as cloud data migration options for your workload. Choosing the right storage service for your workload requires you to make a series of decisions based on your business needs.

### Common Misconceptions

>I absolutely need multi-protocol access to my file shares.

> Mission-critical applications require a [[Storage Area Network (SAN)]] on premises.

> AWS doesn't offer deduplication capability for storage.

> My application is latency-sensitive and won't work in the cloud.

### Decision Criteria
1. Protocol
2. Client type
3. IOPS/access patterns
4. Latency
5. Throughput or bandwidth
6. Geographic scale/data sovereignty
7. Migration strategy and risks
8. Backup/protection requirements
9. Disaster recovery
10. Security
11. Applications (including certifications)
12. Workflow
13. Cost-optimization strategies
14. Cloud-native layered services?

### Common Customer Scenarios

#### Scenario 1: Migrating existing applications to the cloud

1. Existing applications with data on a [[Storage Area Network (SAN)]] or direct-attached storage
2. Existing applications with data on a [[Network Attached Storage (NAS)]] or file share

![[Pasted image 20241028110635.png]]

#### Scenario 2: Backing up data to the cloud

#### Scenario 3: Hybrid - using cloud storage with on-premises applications

#### Scenario 4: Building a data lake

#### Scenario 5: Building a new application

## Streaming Data Services

> Ensure you understand how to use AWS streaming data sources to ingest data, such as [[Amazon Kinesis]], [[Apache Flink]], and [[Apache Kafka]]
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

```
You are using Amazon data Firehose delivery stream to ingest GZIP compressed data records from an on-premises application. You need to configure a solution for your data scientist to perform SQL queries against the data stream for real-time insights.

What is your solution?
```

One solution is to use the AWS Managed Service for [[Apache Flink]] application and a [[AWS Lambda]] function to transform the data before it is processed by the SQL code, and then send the data to the application for real-time analysis.

Another solution might be to store the data in an S3 bucket and use [[Amazon Athena]] to run queries. This would work, but remember, a keyword for this scenario question was real-time. And Athena does not provide real-time insights. Also, Athena cannot consume data directly from the Firehose delivery stream in real-time.

```
Can you migrate data to Amazon S3 using AWS Database Migration Service from an on-premises or other supported database sources?
```

Yes, one solution is to use Amazon S3 as a target in an [[AWS DMS]] task, and both full load and change data capture data is written to comma-separated value format by default. For more compact storage and faster queries, you can use Apache Parquet as the storage format. And after that data is migrated from AWS DMS to Amazon S3, you can use Amazon SageMaker, which has a faster Pipe mode implementation that accelerates the data transfer speeds for data that is streamed from Amazon S3 into SageMaker. This helps your training jobs start sooner, finish quicker, require less disk space, and reduces your cost to train machine learning models on SageMaker.

```
You are a machine learning engineer and you need to process a large amount of customer data, analyze the data, and get insights so that analysts can make further decisions. To accomplish this task, you need to store the data in a data structure that can handle large volumes of data and efficiently retrieve it as fast as possible.

What is your solution?
```

One solution is to use [[Amazon EMR]] with [[Apache Hadoop]] Distributed File System to store the data and then you can process the data to extract insights. You choose the appropriate algorithm, such as the MapReduce algorithm, to process the data in your data processing pipelines. Finally, you optimize the performance of your data processing pipeline by using techniques such as data partitioning, caching, and parallel processing. For this task statement, you also need to understand how to ingest data into [[Amazon SageMaker#Canvas]] and [[Amazon SageMaker#Feature Store]]. In SageMaker, records are added to your feature groups through ingestion, depending on the use case, storage configuration, and more.

Or you can use the [[Amazon SageMaker#Data Wrangler]] experience in Canvas to engineer features and then ingest your features into your [[Amazon SageMaker#Feature Store]]. After the feature group has been created, you can also select and join data across multiple feature groups to create new engineered features in SageMaker Canvas and then export your dataset to an S3 bucket. Is there another way to bring your data into Feature Store? Yes, you can also use Amazon EMR for batch data ingestion through an [[Apache Spark]] connector.
# 1.2: Transform Data and Perform Feature Engineering