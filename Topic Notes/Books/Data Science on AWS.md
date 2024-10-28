By #ChrisFregly & #AntjeBarth

# Chapter 1: Introduction to Data Science on AWS

## Benefits of Cloud Computing

* Agility
	* Spin up resources as needed
* Cost Savings
	* Trade capital expenses for variable expenses
* Elasticity
	* Automatically scale resources up or down to match needs
* Innovate Faster
	* Allows us to focus on developing applications that differentiate our business, rather than the undifferentiated heavy lifting of managing infrastructure.
* Deploy Globally in Minutes
	* *Region* is a physical location around the world where data centers are clustered. *Availability Zones* is a group of logical data centers. Each *Region* contains multiple, isolated, and physically separate *Availability Zones* within a geographic area.
* Smooth Transition from Prototype to Production

## Data Science Pipelines and Workflows

*Data Preparation*
Data Ingestion >> Data Analysis >> Data Transformation >> Data Validation >> Create Training Data >>

*Model Training and Tuning*
Model Training >> Model Tuning >>

*Deployment and Monitoring*
Deployment >> Serving >> Monitoring >> Logging

### [[Amazon SageMaker#SageMaker Pipelines]]

### [[AWS Step Functions]]

### [[TensorFlow Extended (TFX)]]

### [[Topic Notes/MLOps/Data Flows/Human-in-the-loop Workflows|Human-in-the-loop Workflows]]

## MLOps Best Practices

MLOps emerged to describe the unique challenges of operating "software plus data" systems like AI/ML. With MLOps, we are developing the end-to-end architecture for automated model training, model hosting, and pipeline monitoring.
#### 3 Stages of Maturity of MLOps
1. MLOps v1.0
Manually build, train, tune and deploy models.
2. MLOps v2.0
Manually build and orchestrate model pipelines.
3. Automatically run pipelines when new data arrives or code changes from deterministic triggers such as GitOps or when models start to degrade in performance based on statistical triggers such as drift, bias, and explainability divergence.

### Operational Excellence 

 1. Data Quality Checks

Poor quality data leads to many failed projects. Stay ahead of these issues early in the pipeline

 2. Start simple and reuse existing solutions

Start with the simplest solution, no need to reinvent the wheel unnecessarily. Leverage existing managed services like [[Amazon SageMaker]].

 3. Define model performance metrics

Map metrics to business objectives, and continuously monitor these metrics. There should be a strategy to trigger model invalidations and retrain models when performance degrades.

4. Track and version everything

Track model development trough experiments and lineage tracking. We should also version our datasets, feature-transformation code, hyper-parameters, and trained models.

5. Select appropriate hardware for both model training and model serving

Often, model training has different infrastructure requirements than does model-prediction serving.

6. Continuously monitor deployed models

Detect data drift and model drift, and take appropriate action such as model retraining.

7. Automate machine learning workflows

Build consistent, automated pipelines to reduce human error and free up time to focus on the hard problems. Pipelines can include human-approval steps for approving models before pushing them to production.

### Security

>Security and compliance is a shared responsibility between AWS and the customer

AWS ensures the security "*of*" the cloud, while the customer is responsible for security "*in*" the cloud. The most common security considerations:
* Access Management
* Compute and Network Isolation
* Encryption
* Governance
* Auditability

### Reliability

The ability of a system to recover from infrastructure or service disruptions, acquire computing resources dynamically to meet demand, and mitigate disruptions such as misconfigurations or transient network issues.

We should automate change tracking and versioning for our training data. This way, we can re-create the exact version of a model in the event of a failure. We will build once and use the model artifacts to deploy the model across multiple AWS accounts and environments.

### Performance Efficiency

Refers to the efficient use of computing resources to meet requirements and how to maintain that efficiency as demand changes and technologies evolve. For example, we can use GPU-based instances to more efficiently train deep learning models using a larger queue depth, higher arithmetic logic units, and increased register counts.

Know the latency and network bandwidth performance requirements of models, and deploy each model closer to customers, if needed. There are situations where we might want to deploy our models *at the edge* to improve performance or comply with data-privacy regulations. *Deploying at the edge* refers to running the model on the device itself to run the predictions locally. We also want to continuously monitor key performance metrics of our model to spot performance deviations early.

### Cost Optimization

We can optimize cost by leveraging different [[Amazon EC2]] instance pricing options. For example, Savings Plans offer significant savings over on-demand instance prices, in exchange for a commitment to use a specific amount of compute power for a given amount of time. Savings Plans are a great choice for known/steady state workloads such as stable inference workloads.

For on-demand instances, we pay for compute capacity by the hour or second demanding on the instance. These are best for new or stateful spiky workloads such as short-term model training jobs.

See [[Amazon EC2#Spot Instances]]
## Amazon AI Services and AutoML with [[Amazon SageMaker]]

"*AutoML*" commonly refers to the effort of automating the typical steps of a model development workflow.
## Data Ingestion, Exploration, and Preparation in AWS

Covered in more detail in [[Data Science on AWS#Chapter 4 Ingest Data into the Cloud]], [[Data Science on AWS#Chapter 5 Explore the Dataset]], and [[Data Science on AWS#Chapter 6 Prepare the Dataset for Model Training]].

### Data Ingestion and Data Lakes with [[Amazon S3]] and [[AWS Lake Formation]]

>Business success is often closely related to a company's ability to quickly extract value from their data.

This is why many companies are moving to a highly scalable, available, secure, and flexible data store, often called a *data lake*.

A *data lake* is a centralized and secure repository that enables us to store, govern, discover, and share data at any scale. With a data lake, we can run any kind of analytics efficiently, and use multiple AWS services without having to transform and deploy more accurate models.

We can use the [[AWS Lake Formation]] service to create data lakes. We can leverage [[AWS Glue]] to automatically discover and profile new data. This ad hoc exploration and prototyping can be done from development environments such as [[Amazon SageMaker#Studio]], [[AWS Glue#DataBrew]], and [[Amazon SageMaker#Data Wrangler]].

### Data Analysis with [[Amazon Athena]], [[Amazon Redshift]], and [[Amazon QuickSight]]

In the data analysis step, we explore our data, collect statistics, check for missing values, calculate quantiles, and identify data correlations. We call this "*ad hoc*" exploration and prototyping, where we query parts of our data to get a first understanding of the data schema and data quality for our specific machine learning problem at hand. We then develop model code and ensure it is functionally correct.

Note that libraries like [[Pandas]] use in-memory data structures (*DataFrames*) to hold and manipulate data, so we need to be careful how much is loaded at once.

### Evaluate Data Quality with [[AWS Deequ]] and [[Amazon SageMaker#Processing Jobs]]

We need high-quality data to build high-quality models. Before we create our training dataset, we want to ensure our data meets certain quality constraints. In software development, we run unit tests to ensure our code meets design and quality standards and behaves as expected. Similarly, we can run unit tests on our dataset to ensure the data meets our quality expectations.

### Label Training Data with [[Amazon SageMaker#Ground Truth]]

Many data science projects implement supervised learning, where models learn by example. We first need to collect and evaluate, then provide accurate labels. If there are incorrect labels, our ML model will learn from bad examples.
### Data Transformation with [[AWS Glue#DataBrew]], [[Amazon SageMaker#Data Wrangler]], and [[Amazon SageMaker#Processing Jobs]].

Assuming we have our data in an [[Amazon S3]] datalake, or S3 bucket, the next step is to prepare the data for model training. Data transformations might include dropping or combining data in our dataset. We might need to convert text data into word embeddings for use with natural language models. Or perhaps we might need to convert data into another format, from numerical to text representations, or vice versa.
## Model Training and Tuning with [[Amazon SageMaker]]

Here we explore model training and tunings steps of our model development workflow in more detail and learn which AWS services and open source tools we can leverage.

## Model Deployment with [[Amazon SageMaker]] and [[AWS Lambda]] Functions

## Streaming Analytics and Machine Learning on AWS

## AWS Infrastructure and Custom-Built Hardware

## Reduce Cost with Tags, Budgets, and Alerts

## Summary

# Chapter 2: Data Science Use Cases

# Chapter 3: Automated Machine Learning

# Chapter 4: Ingest Data into the Cloud

# Chapter 5: Explore the Dataset

# Chapter 6: Prepare the Dataset for Model Training

# Chapter 7: Train Your First Model

# Chapter 8: Train and Optimize Models at Scale

# Chapter 9: Deploy Models to Production

# Chapter 10: Pipelines and MLOps

# Chapter 11: Streaming Analytics and Machine Learning

# Chapter 12: Secure Data Science on AWS



