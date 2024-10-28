Amazon Simple Storage Service (Amazon S3) is an object storage service that offers industry-leading scalability, data availability, security, and performance. Customers of all sizes and industries can use Amazon S3 to store and protect any amount of data for a range of use cases, such as data lakes, websites, mobile applications, backup and restore, archive, enterprise applications, IoT devices, and big data analytics. Amazon S3 provides management features so that you can optimize, organize, and configure access to your data to meet your specific business, organizational, and compliance requirements.

Designed for 99.9...9% (11 nines) of durability as well as for strong read-after-write consistency. S3 is a popular choice for data lakes in AWS.

## Buckets

There are two types of Amazon S3 buckets, general purpose buckets and directory buckets
* **General Purpose Buckets** are the original S3 bucket type and are recommended for most use cases and access patterns.
* **Directory Buckets** use the S3 Express One Zone storage class, which is recommended if your application is performance sensitive and benefits from single-digit millisecond `PUT` and `GET` latencies.

Directory buckets are used for workloads or performance-critical applications that require consistent single-digit millisecond latency. Directory buckets organize data hierarchically into directories as opposed to the flat storage structure of general purpose buckets. There aren't prefix limits for directory buckets, and individual directories can scale horizontally.

Directory buckets use the S3 Express One Zone storage class, which stores data across multiple devices within a single Availability Zone but doesn't store data redundantly across Availability Zones. When you create a directory bucket, we recommend that you specify an AWS Region and an Availability Zone that's local to your Amazon EC2, Amazon Elastic Kubernetes Service, or Amazon Elastic Container Service (Amazon ECS) compute instances to optimize performance.

You can create up to 10 directory buckets in each of your AWS accounts, with no limit on the number of objects that you can store in a bucket.
## Glacier

Use Amazon S3 Glacier for data archiving use cases and the low cost archive storage in the cloud.

## S3 on Outposts

Use Amazon S3 on Outposts for local data processing and data residency use cases in your on-premises [[AWS Outposts]] environments.
## Storage Classes

## Storage Lens

## Lifecycle Policies

## Security

See more in [[Data Science on AWS#Chapter 12 Secure Data Science on AWS]]

