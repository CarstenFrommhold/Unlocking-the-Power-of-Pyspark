# Unlocking the Power of PySpark

## tl;dr

This repo aims to cover the key concepts of Apache Spark and it's Python API, Pyspark.
Apache Spark is an open-source unified analytics engine for large-scale data processing. 
It consists of five components, namely *Core*, *SQL*, *ML*, *Streaming* and *GraphX*.
Here, the first three are touched.

## What to expect

* First, we will get to know the basic principles of Spark and its Python API, PySpark.
* Next, we'll take a look at Spark's machine learning capabilities. 
* With a business case in mind, we turn away from the local setup and move to the cloud, namely AWS.

## Setup

To be able to start with the content, there are several options.

Option 1)   
You can build a virtual environment with the corresponding dependencies via
poetry or requirements.

```bash
poetry install
jupyter lab
```

Option 2)  
As an alternative, you can start it directly via docker.


``` bash
docker build -t tutorial .
```

``` bash
docker run -p 8888:8888 tutorial
```

Option 3)  
Do it directly in Google Colab.

* [Principles Notebook](https://colab.research.google.com/drive/1bR6bSXXpiPCEUI6MIXy6LCsFT7dS6fQq?usp=sharing)  
* [ML Notebook](https://colab.research.google.com/drive/1Hw83ROwYTioPq3iSEHyqSJmOanijWZ77?usp=sharing)  
* [Cloud Motivation](https://colab.research.google.com/drive/187vFM9ROGl091RCYz4jixYotdXCFu-2C?usp=sharing)  



## Is this really Python?

PySpark is a Python API for Apache Spark. 
While Spark is implemented in Scala and runs on the Java Virtual Machine (JVM), PySpark allows Python developers to interface with Spark and take advantage of its distributed computing capabilities.
Besides Pyspark, however, all other libraries of Python can be used and combined in your scripts. 

### So do i need Scala to be installed?

No, you don't need to install Scala to use PySpark. PySpark comes bundled with a pre-built version of Spark that includes the Scala runtime environment. When you use PySpark, you don't interact with Scala directly, but PySpark communicates with the Spark runtime environment, which is implemented in Scala and runs on the Java Virtual Machine (JVM). So, you need Java to be installed.

## How can I imagine a cluster figuratively?

Here you are.

![Spark in cluster mode](https://spark.apache.org/docs/latest/img/cluster-overview.png)

## References

* [Apache Spark Docs](https://spark.apache.org/docs/latest/)
* [Lazy Evaluation in Apache Spark – A Quick guide](https://data-flair.training/blogs/apache-spark-lazy-evaluation/)
* [6 recommendations for optimizing a Spark job](https://towardsdatascience.com/6-recommendations-for-optimizing-a-spark-job-5899ec269b4b)
* [Scalable Machine Learning with Spark](https://towardsdatascience.com/scalable-machine-learning-with-spark-807825699476)
* [Easy Local PySpark Environment Setup Using Docker](https://medium.com/@antoniolui/easy-5-minute-local-pyspark-environment-setup-using-docker-e9c53c0f3b84)