Amazon EBS (Elastic Block Store) is a block storage services provided by AWS designed for use within [[Amazon EC2]] instances. EBS provides high-performance, persistent storage volumes that can be attached to EC2 instances, allowing applications to access data at the block level, similar to a traditional hard drive.

EBS is highly reliable and efficient for workloads needing consistent, high-performance storage tightly integrated with Amazon EC2, making it popular for a wide variety of enterprise and cloud-native applications.
## Key Features

1. **Block-Level Storage**: EBS operates at the block level, meaning data is stored in fixed-size blocks, making it ideal for applications requiring low-latency and high-throughput storage (e.g., databases).
2. **Persistence**: EBS volumes are persistent, meaning they retain data even when the EC2 instance they're attached to is stopped or terminated.
3. **Customizable Performance**
	1. *SSD-Backed Volumes*: Optimized for transactional workloads, with high IOPS (input/output operations per second), suitable for databases and other performance-sensitive applications.
	2. *HDD-Backed Volumes*: Optimized for throughput-intensive applications that handle large, sequential data sets, such as big data and log processing.
4. **Snapshot and Backup Capabilities**: EBS provides the ability to create snapshots, which are incremental backups of the volume stored in [[Amazon S3]]. These snapshots can be used for data backup, disaster recovery, and creating new volumes.
5. **Scalability and Flexibility**: EBS volumes can scale in size, performance, and type to meet the needs of different applications. They can be resized or upgraded with minimal downtime.
6. **Encryption**: EBS supports data encryption at rest and in transit, using [[AWS KMS]] for enhanced security.
7. **Availability and Durability**: EBS volumes are designed for high availability and durability within a single Availability Zone, but snapshots can be replicated across regions for disaster recovery.
## Common Use Cases

1. **Databases**: Storing data for relational and NoSQL databases that require high IOPS and low latency
2. **File Systems**: Hosting file systems that applications and users access as if they were local drives.
3. **Enterprise Applications**: Supporting applications that need high availability, like SAP or ERP systems.
4. **Backup and Disaster Recovery**: Using snapshots for regular data backup and recovery.

## Volume Types

Tradeoffs between IOPS, Latency, Cost, Performance, and Throughput
### io2 Block Express

### I3

### gp3

### io2

### sc1

### st1

### D3