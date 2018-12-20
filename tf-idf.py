import pyspark
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Find how many files are in the location

files = 100

def main(sc):
    

    # Load file into rdd
    textFile = sc.sparkContext.wholeTextFiles("hdfs:/user/root/ds8003/exam/*.txt")
    
    # flatten by values (since we have a tuple of file location, word)
    wordList = textFile.flatMapValues(lambda line: line.split())
    # Map and reduce
    wordCount = wordList.map(lambda word: (word,1))
    wordWithTotalCount = wordCount.reduceByKey(lambda v1, v2: v1+v2)
    
    # Convert to dataframe and separate the tuple into three distinct columns
    df = wordWithTotalCount.map(lambda x: Row(file=x[0][0],word=x[0][1],cnt=x[1])).toDF()
    
    # Calculate TF  
    sqlContext = SQLContext(sc)
    df.registerTempTable("df")
    df = sqlContext.sql("SELECT substring(file,-7,7) as file, word, sum(cnt) as cnt, (sum(cnt)/sum(sum(cnt)) over (partition by file)) as tf FROM df group by file, word")

    # Create the inverse index master file
    inv_index = df.groupBy("word").agg(F.collect_list("file").alias("files"),F.sum("cnt").alias("freq"), F.count("file").alias("num_docs"))

    inv_index.write.parquet("hdfs:/user/root/ds8003/exam/inv_index.parquet")

    # Calculate IDF and TF-IDF
    docs = df.join(inv_index, (df.word == inv_index.word)).select(df.file,df.word,df.cnt,df.tf,F.log((files/inv_index.num_docs)).alias("idf"),(df.tf*F.log((files/inv_index.num_docs))).alias("tf_idf"))

    # Create a parquet file
    docs.write.parquet("hdfs:/user/root/ds8003/exam/docs.parquet")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Test").config(conf = SparkConf()).getOrCreate()
    main(spark)
    spark.stop()
