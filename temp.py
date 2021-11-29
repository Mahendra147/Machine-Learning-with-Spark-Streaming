import pyspark
import numpy as np
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
#from pyspark.ml import Pipeline
from pyspark.sql import Row,SQLContext,SparkSession
#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
#from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
#from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
from sparknlp.base import *
from sparknlp.annotator import *
#from sparknlp import DocumentAssembler

from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

from pyspark.ml.feature import Tokenizer



if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="trial")
	ssc = StreamingContext(sc,5)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	

	word=lines.flatMap(lambda line: line.split("\n"))
	#word=word.map(lambda lines: json.loads(lines))
	def cnf(rd):
		f0=[]
		f1=[]
		f2=[]
		#print(rd)
		df= spark.read.json(rd)
		#df=sqlContext.createDataFrame(rd)
		f=df.collect()
		for i in f:
			for k in i:
				f0.append(k[0])
				f1.append(k[1])
				f2.append(k[2])
		if(len(f0)!=0 and len(f1)!=0 and len(f2)!=0):
			x=sql.createDataFrame(zip(f0,f1,f2),schema=['Subject','Email_content','label'])
			x.show()
			#print(x)
			#print(df.collect())
			#print(f0)
			#x=spark.read.json(rd)
			#sqlContext.implicits._rdd.toDf()
			#=x.rdd.map(lambda a: (lambda b: [b[0],b[1],b[2]]))
			#print(y.collect())
			documentAssembler = DocumentAssembler()\
    			.setInputCol("Subject")\
    			.setOutputCol("assembled_sub")\
    			.setCleanupMode("shrink")

			sentenceDetector = SentenceDetector()\
    			.setInputCols(['assembled_sub'])\
    			.setOutputCol('sentenced_sub')

			"""tokenizer = Tokenizer() \
    			.setInputCol("sentenced_sub") \
    			.setOutputCol("tokenised_sub")"""
    			
			tokenizer = Tokenizer(inputCol="sentenced_sub", outputCol="tokenised_sub")
    			

			normalizer = Normalizer() \
    			.setInputCols(["tokenised_sub"]) \
    			.setOutputCol("normalised_sub")\
    			.setLowercase(False)

			stopwords_cleaner = StopWordsCleaner()\
    			.setInputCols(["normalized_sub"])\
	    		.setOutputCol("cleaned_sub")\
	    		.setCaseSensitive(False)\

			tokenassembler = TokenAssembler()\
	    		.setInputCols(["sentenced_sub", "cleaned_sub"]) \
	    		.setOutputCol("tk_sub")


			nlpPipeline = Pipeline(stages=[
	    			documentAssembler,
	    			sentenceDetector,
	    			tokenizer,
	    			#normalizer,
	    			#stopwords_cleaner,
	    			#tokenassembler
	    			])
	    		
	    		
			pipelineModel = nlpPipeline.fit(x)

			result = pipelineModel.transform(x)

			result.show()
	
	
	rdd=word.foreachRDD(cnf)
	#rdd.pprint()
	#rdd=word.map(lambda x: json.loads(x))	
	#r=json.loads(lines)
	print("new batch")
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()

