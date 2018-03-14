from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import *

spark = SparkSession\
  .builder\
  .appName("hackathon")\
  .getOrCreate()
    
data = spark.read.csv("data.csv", header=True, mode="DROPMALFORMED", inferSchema=True)

# transform the data
assembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
assembled = assembler.transform(data).select('label', 'features')

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(assembled)

# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

def question(x, y):
  cSchema = StructType([StructField("x", FloatType()), StructField("y", FloatType())])
  questionDF = spark.createDataFrame([[float(x), float(y)]], schema=cSchema)
  lrModel.transform(assembler.transform(questionDF)).select('prediction').show()
