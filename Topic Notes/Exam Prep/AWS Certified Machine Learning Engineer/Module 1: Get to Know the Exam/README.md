# Get to know the exam with exam-styled questions

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

### 3. Deployment and Orchestration of ML Workflows: 22% of the Exam

### 4. ML Solution Monitoring, Maintenance, and Security: 24% of the Exam

## AWS Services for Learning

- AWS Builder labs
- AWS Cloud Quest
- AWS Jam
