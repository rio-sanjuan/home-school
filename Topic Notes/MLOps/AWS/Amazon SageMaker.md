Amazon SageMaker offers us a hosted managed Jupyter environment and an integrated development environment with [[Amazon SageMaker#Studio]].
## Pipelines

SageMaker Pipelines are the standard, full-featured, and most complete way to implement AI/ML pipelines in SageMaker.
## Studio

## Canvas
## Feature Store

## Data Wrangler

SageMaker Data Wrangler offers low-code, UI-driven data transformations. We can read data from various sources, including [[Amazon S3]], [[Amazon Athena]], [[Amazon Redshift]], and [[AWS Lake Formation]]. SageMaker Data Wrangler comes with pre-configured data transformations similar to [[AWS Glue#DataBrew]] to convert column types, perform one-hot encoding, and process text fields. SageMaker Data Wrangler supports custom user-defined functions using [[Apache Spark]] and even generates code including Python scripts and SageMaker Processing Jobs.
## Processing Jobs

SageMaker Processing Jobs let us run custom data processing code for data transformation, data validation, or model evaluation across data in [[Amazon S3]]. When we configure the processing job, we define the resources needed, including instance types and number of instances. SageMaker takes our custom code, copies our data from S3, and then pulls a Docker container to execute the processing step.
## Ground Truth

SageMaker Ground Truth helps us to efficiently and accurately label data stored in [[Amazon S3]]. Ground Truth uses a combination of automated and human data labeling. Ground Truth provides pre-built workflows and interfaces for common data labeling tasks. We define the labeling task and assign the labeling job to either a public workforce via [[Amazon Mechanical Turk]] or a private workforce, such as our coworkers. We can also leverage third-party data labeling service providers listed on the AWS Marketplace, which are prescreened by Amazon.

Ground Truth implements active learning techniques for pre-built workflows. It creates a model to automatically label a subset of the data, based on the labels assigned by the human workforce. As the model continuously learns from the human workforce, the accuracy improves, and less data needs to be sent to the human workforce. Over time and with enough data, the SageMaker Ground Truth active-learning model is able to provide high-quality and automatic annotations that result in lower labeling costs overall.

See more in [[Data Science on AWS#Chapter 10 Pipelines and MLOps]]
## Training Jobs

## Hyper-Parameter Tuning Jobs

## Model Registry

## Batch Transform

## Model Endpoints