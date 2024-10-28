By #JulieElkins

Exam results are reported between 100-1,000. A passing score is 720.
## Domain Overview

### 1. Data Preparation for Machine Learning (ML): 28% of the Exam

#### 1.1: Ingest and Store Data

Knowledge of:

- Data formats and ingestion mechanisms (parquet, json, csv, etc.)
- How to use the core AWS data sources (s3, EFS, etc.)
- How to use AWS streaming data sources to ingest data (kinesis, flink, kafka, etc.)
- AWS storage option, including use cases and tradeoffs

Skills in:

- Extracting data from storage (s3, EBS, EFS, RDS, DynamoDB, etc.) by using relevant AWS service options (s3 transfer acceleration, EBS provisioned IOPS)
- Choosing appropriate data formats based on data access patterns
- Ingesting data into SageMaker Data Wrangler and SageMaker Feature Store
- Merging data from multiple sources (e.g. by using programming techniques, AWS Glue, Apache Spark)
- Troubleshooting and debugging data ingestion and storage issues that involve capacity and scalability
- Making initial storage decisions based on cost, performance, and data structure

#### 1.2: Transform Data and Perform Feature Engineering

Knowledge of:

- Data cleaning and transofmration techniques (e.g., detecting and treating outliers, imputing missing data, combing, deduplication)
- Feature engineering techniques (e.g., data scaling and standardization, feature splitting, binning, log transofmration, normalization)
- Encoding techniques (e.g., one-hot encoding, binary encoding, label encoding, tokenization)
- Tools to explore, visualize, or transform data and features (e.g., SageMaker Data Wrangler, AWS Glue, AWS Glue DataBrew)
- Services that transform streaming data (e.g., AWS Lambda, Spark)
- Data annotation and labeling services that create high-quality labeled datasets

Skills in:

- Transforming data by using AWS tools (e.g., AWS Glue, AWS Glue DataBrew, Spark running on Amazon EMR, SageMaker Data Wrangler)
- Creating and managing features by using AWS tools (e.g., SageMaker Feature Store)
- Validating and labeling data by using AWS services (e.g., Sagemaker Ground Truth, Amazon Mechanical Turk)

#### 1.3: Ensure Data Integrity and Prepare Data for Modeling

Knowledge of:

- Pre-training bias metrics for numeric, text, and image data (e.g., class imbalance (CI), difference in proportion labels (DPL))
- Strategies to address CI in numeric, text, and image datasets (e.g., synthetic data generation, resampling)
- Techniques to encrypt data
- Data classification, anonymization, and masking
- Implications of compliance requirements (e.g., personally identifiable information (PII), protected health information (PHI), data residency)

Skills in:

- Validating data quality (e.g., by using AWS Glue DataBrew, AWS Glue Data Quality)
- Identifying and mitigating sources of bias in data (e.g., selection bias, measurement bias) by using AWS tools (e.g., SageMaker Clarify)
- Preparing data to reduce prediction bias (e.g., by using dataset splitting, shuffling, and augmentation)
- Configuring data to load into the model training resource (e.g., Amazon EFS, Amazon FSx)

### 2. Model Development: 26% of the Exam

#### 2.1: Choose a modeling approach

Knowledge of:

- Capability and appropriate uses of ML algorithms to solve business problems
- How to use AWS artificial intelligence (AI) services (e.g., Amazon Translate, Amazon Transcribe, Amazon Rekognition, Amazon Bedrock) to solve specific business problems
- How to consider interpretability during model selection or algorithm selection
- SageMaker built-in algorithms and when to apply them

Skills in:

- Assessing available data and problem complexity to determine the feasibility of an ML solution
- Comparing and selecting appropriate ML models or algorithms to solve specific problems
- Choosing built-in algorithms, foundation models, and solution templates (e.g. in SageMaker JumpStart and Amazon Bedrock)
- Selecting models or algorithms based on costs
- Selecting AI services to solve common business needs

#### 2.2: Train and Refine Models

Knowledge of:

- Elements in the training process (e.g., epoch, steps, and batch size)
- Methods to reduce model training time (e.g., early stopping, distributed training)
- Methods to improve model performance
- Benefits of regularization techniques (e.g., dropout, weight decay, L1 and L2 regularization)
- Hyperparameter tuning techniques (e.g., random search, Bayesian optimization)
- Model hyperparameters and their effects on model performance (e.g., number of trees in a tree-based model, number of layers in a neural network)
- Methods to integrate models that were built outside SageMaker into SageMaker

Skills in:

- Using SageMaker built-in algorithms and common ML libraries to develop ML models
- Using SageMaker script mode with SageMaker supported frameworks to train models (e.g., TensorFlow, PyTorch)
- Using custom datasets to fine-tune pre-trained models (e.g., Amazon Bedrock, SageMaker JumpStart)
- Performing hyperparameter tuning (e.g., by using SageMaker automatic model tuning (AMT))
- Integrating autoamted hyperparameter optimization capabilities
- Preventing model overfitting, underfitting, and catastrophic forgetting (e.g., by using regularization techniques, feature selection)
- Combining multiple training models to improve performance (e.g., ensembling, stacking, boosting)
- Reducing model size (e.g., by altering data types, pruning, updating feature selection, compression)
- Managing model versions for repeatabliity and audits (e.g., by using SageMaker Model Registry)

#### 2.3: Analyze Model Performance

Knowledge of:

- Model evaluation techniques and metrics (e.g., confusion matrix, heat maps, F1 score, accuracy, precision, recall, Root Mean Square Error (RMSE), Receiver Operating Characteristic (ROC), Area Under the ROC Curve (AUC))
- Methods to create performance baselines
- Methods to identify model overfitting and underfitting
- Metrics available in SageMaker Clarify to gain insights into ML training data and models
- Convergence issues

Skills in:

- Selecting and interpreting evaluation metrics and detecting model bias
- Assessing tradeoffs between model performance, training time, and cost
- Performing reproducible experiments by using AWS services
- Comparing the performance of a shadow variant to the performance of a production variant
- Using SageMaker Clarify to interpret model outputs
- Using SageMaker Model Debugger to debug model convergence

### 3. Deployment and Orchestration of ML Workflows: 22% of the Exam

#### 3.1: Selecting Deployment Infrastructure based on Existing Architecture and Requirements

Knowledge of:

- Deployment best practices (e.g., versioning, rollback strategies)
- AWS deployment services (e.g., SageMaker)
- Methods to serve ML mdoels in real time and in batches
- How to provision compute resources in production environments and test environments (e.g., CPU, GPU)
- Model and endpoint requirements for deployment endpoints (e.g., serverless endpoints, real-time endpoints, asynchronous endpoints, batch inference)
- How to choose appropriate containers (e.g., provided or customized)
- Methods to optimize models on edge devices (e.g., SageMaker Neo)

Skills in:

- Evaluating performance, cost, and latency tradeoffs
- Choosing the appropriate compute environment for training and inference based on requirements (e.g., GPU or CPU specifications, processor family, networking bandwidth)
- Selecting the correct deployment orchestrator (e.g., Apache Airflow, SageMaker Pipelines)
- Selecting multi-model or multi-container deployments
- Selecting the correct deployment target (e.g., SageMaker endpoints, Kubernetes, Amazon Elastic Container Service (ECS), Amazon Elastic Kubernetes Service (EKS), AWS Lambda)
- Choosing model deployment strategies (e.g., real time, batch)

#### 3.2: Create and Script Infrastructure Based on Existing Architecture and Requirements

Knowledge of:

- Difference between on-demand and provisioned resources
- How to compare scaling policies
- Tradeoffs and use cases of infrastructure as code (IaC) options (e.g., AWS CloudFormation, AWS Cloud Development Kit (AWS CDK))
- Containerization concepts and AWS container services
- How to use SageMaker endpoint auto scaling policies to meet scalability requirements (e.g., based on demand, time)

Skills in:

- Applying best practices to enable maintainable, scalable, and cost-effective ML solutions (e.g., automatic scaling on SageMaker endpoints, dynamically adding Spot instances, by using Amazon EC2 instances, by using Lambda behind the endpoints)
- Automating the provisioning of compute resources, including communication between stacks (e.g., by using CloudFormation, AWS CDK)
- Building and maintaining containers (e.g., Amazon Elastic Container Registry (ECR), Amazon EKS, Amazon ECS, by using bring your own container (BYOC) with SageMaker)
- Configuring SageMaker endpoints within the VPC network
- Deploying and hosting models by using the SageMaker SDK
- Choosing specific metrics for auto scaling (e.g., model latency, CPU utilization, invocations per instance)

#### 3.3 Use Automated Orchestration Tools to Set Up Continuous Integration and Continuous Delivery (CI/CD) pipelines.

Knowledge of:

- Capabilities and quotas for AWS CodePipeline, AWS CodeBuild, and AWS CodeDeploy
- Automation and integratino of data ingestion with orchestration services
- Version control systems and basic usage (e.g., Git)
- CI/CD principles and how they fit into ML workflows
- Deployment strategies and rollback actions (e.g., blue/green, canary, linear)
- How code repositories and pipelines work together

Skills in:

- Configuring and troubleshooting CodeBuild, CodeDeploy, and CodePipeline, including stages
- Applying continuous deployment flow structures to invoke pipelines (e.g., Gitflow, GitHub Flow)
- Using AWS services to automate orchestration (e.g., to deploy ML models, automate model building)
- Configuring training and inference jobs (e.g., by using Amazon EventBridge rules, SageMaker piplines, CodePipeline)
- Creating automated tests in CI/CD pipelines (e.g., integration tests, unit tests, end-to-end tests)
- Building and integrating mechanisms to retrain models

### 4. ML Solution Monitoring, Maintenance, and Security: 24% of the Exam

#### 4.1: Monitor Model Inference

Knowledge of:

- Drift in ML Models
- Techniques to monitor data quality and model performance
- Design principles for ML lenses relevant to monitoring

Skills in:

- Monitoring models in production (e.g., by using SageMaker Model Monitor)
- Monitoring workflows to detect anomalies or errors in data processing or model inference
- Detecting changes in the distribution of data that can affect model performance (e.g., by using SageMaker Clarify)
- Monitoring model performance in production by using A/B testing

#### 4.2: Monitor and Optimize Infrastructure and Costs

Knowledge of:

- Key performance metrics for ML infrastructure (e.g., utilization, throughput, availability, scalability, fault tolerance)
- Monitoring and observability tools to troubleshoot latency and performance issues (e.g., AWS X-Ray, Amazon CloudWatch Lambda Insights, Amazon CloudWatch Logs Insights)
- How to use AWS CloudTrail to log, monitor, and invoke re-training activities
- Differences between instance types and how they affect performance (e.g., memory optimized, compute optimized, general purpose, inference optimized)
- Capabilities of cost analysis tools (e.g., AWS Cost Explorer, AWS Billing and Cost Management, AWS Trusted Advisor)
- Cost tracking and allocation techniques (e.g., resource tagging)

Skills in:

- Configuring and using tools to troubleshoot and analyze resources (e.g., CloudWatch Logs, CloudWatch alarms)
- Creating CloudTrail trails
- Setting up dashboards to monitor performance metrics (e.g., by using Amazon QuickSight, CloudWatch dashboards)
- Monitoring infrastructure (e.g., by using EventBridge events)
- Rightsizing instance families and sizes (e.g., by using SageMaker Inference Recommender and AWS Compute Optimizer)
- Monitoring and resolving latency and scaling issues
- Preparing infrastructure for cost monitoring (e.g., by applying a tagging strategy)
- Troubleshooting capacity concerns that involve cost and performance (e.g., provisioned concurrency, service quotas, auto scaling)
- Optimizing costs and setting cost quotas by using appropriate cost management tools (e.g., AWS Cost Explorer, AWS Trusted Advisor, AWS Budgets)
- Optimizing infrastructure costs by selecting purchasing options (e.g., Spot Instances, On-Demand Instances, Reserved Instances, SageMaker Savings Plans)

#### 4.3: Secure AWS Resources

Knowledge of:

- IAM roles, policies, and groups that control access to AWS services (e.g., AWS Identity and Access Management (IAM), bucket policies, SageMaker Role Manager)
- SageMaker security and compliance features
- Controls for network access to ML resources
- Security best practices for CI/CD pipelines

Skills in:

- Configuring least privilege access to ML artifacts
- Configuring IAM policies and roles for users and applications that interact with ML systems
- Monitoring, auditing, and logging ML systems to ensure continued security and compliance
- Troubleshooting and debugging security issues
- Building VPCs, subnets, and security groups to securely isolate ML systems
