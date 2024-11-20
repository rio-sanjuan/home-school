
A SAN is a specialized, high-speed network that provides block-level storage to servers. It is designed to enable multiple servers to access storage devices, such as disk arrays or tape libraries, as if they were directly attached to their own storage. Unlike [[Network Attached Storage (NAS)]], which operates at the file level, SANs operate at the block level, making them ideal for high-performance applications like databases and virtualization.

SANs are often used in data centers where performance, reliability, and scalability are prioritized, although cloud-based environments like AWS use similar underlying principles with services like [[Amazon EBS]] for block storage, which emulates some SAN characteristics within the cloud.
## Key Characteristics

1. **High Performance**: Offers fast data transfer speeds with low latency, as it connects storage directly to servers over a dedicated network.
2. **Reliability and Redundancy**: Often designed with built-in redundancy and failover capabilities to support critical applications.
3. **Flexibility and Scalability**: Can scale to support large amounts of storage and connect multiple servers, providing centralized and flexible storage management.