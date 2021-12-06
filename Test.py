import pyspark
import numpy as np
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

import joblib

from sklearn.linear_model import Perceptron
from pyspark.sql import Row,SQLContext,SparkSession

from pyspark.sql import Row
from sparknlp.base import *
from sparknlp.annotator import *
#from sparknlp import DocumentAssembler

from sklearn import linear_model
from pyspark.ml.linalg import Vector

from pyspark.ml.feature import *
#from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="trial")
	ssc = StreamingContext(sc,5)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	

	word=lines.flatMap(lambda line: line.split("\n"))
	def main(rd):
		arg0=[]
		arg1=[]
		arg2=[]
		df= spark.read.json(rd)
		f=df.collect()
		for i in f:
			for k in i:
				arg0.append(str(k[0]))
				arg1.append(str(k[1]))
				arg2.append(str(k[2]))
		if(len(arg0)!=0 and len(arg1)!=0 and len(arg2)!=0):
			x=sql.createDataFrame(zip(arg0,arg1,arg2),schema=['Subject','Email_content','label'])
			#x.show()
			
			
			rt=RegexTokenizer(inputCol="Subject",outputCol="trial",pattern="\\W+")
			
			stopremove = StopWordsRemover(inputCol='trial',outputCol='stop_tokens')
			
			cv = CountVectorizer(inputCol="stop_tokens", outputCol="token_features", minDF=2.0)
			
			rt1=RegexTokenizer(inputCol="Email_content",outputCol="trial1",pattern="\\W+")
			
			stopremove1 = StopWordsRemover(inputCol='trial1',outputCol='stop_tokens1')
			
			
			bigchunks = NGram().setN(2).setInputCol('stop_tokens').setOutputCol('bigchunks')
			
			bigchunks1 = NGram().setN(2).setInputCol('stop_tokens1').setOutputCol('bigchunks1')
    			
			ht = HashingTF(inputCol="bigchunks", outputCol="hastb",numFeatures=8000)
			
			ht1 = HashingTF(inputCol="bigchunks1", outputCol="hastb1",numFeatures=8000)
			
			label_index = StringIndexer(inputCol="label",outputCol="label_index")
			
			
			nlpPipeline = Pipeline(stages=[
	    			rt1,
	    			stopremove1,
	    			label_index,
	    			bigchunks1,
	    			ht1
	    			])
	    		
			pipelineModel = nlpPipeline.fit(x)

			result = pipelineModel.transform(x)
			X = np.array(result.select("hastb1").collect())
			Y = np.array(result.select("label_index").collect())
			
			nsamples, nx, ny = X.shape
			X = X.reshape((nsamples,nx*ny))
			def bernoullinb(X,Y):
			
				try:
					buildmodel = joblib.load('/home/pes1ug19cs285/bernoullimodel.pkl')
					temp=buildmodel.predict(X)
					joblib.dump(buildmodel, '/home/pes1ug19cs285/bernoullimodel.pkl')
					print(accuracy_score(temp,Y))
				except:
					#EXCEPTION RAISED
					pass
			def sgd(X,Y):
				try:
					buildmodel = joblib.load('/home/pes1ug19cs285/linearmodel_SGD.pkl')
					temp=buildmodel.predict(X)
					joblib.dump(buildmodel,'/home/pes1ug19cs285/linearmodel_SGD.pkl')
					print(accuracy_score(temp,Y))
				except Exception as e:
					#EXCEPTION RAISED
					pass
			def perceptron(X,Y):		
				try:
					buildmodel = joblib.load('/home/pes1ug19cs285/perceptron.pkl')
					temp=buildmodel.predict(X)
					joblib.dump(buildmodel, '/home/pes1ug19cs285/perceptron.pkl')
					print(accuracy_score(temp,Y))
				except Exception as e:
					#EXCEPTION RAISED
					return
			def clustering(X,Y):
				try:
					
					buildmodel = joblib.load('/home/pes1ug19cs285/clustering.pkl')
					temp=buildmodel.predict(X)
					joblib.dump(buildmodel, '/home/pes1ug19cs285/clustering.pkl')
					print(accuracy_score(temp,Y))
				except Exception as e:
					print("BLAH")
					#kmeans = MiniBatchKMeans(n_clusters=2, random_state=0,max_iterations=5)
					#kmeans.partial_fit(X)
					#joblib.dump(newmodel, '/home/pes1ug19cs285/perceptron.pkl')
			
			bernoullinb(X,Y)
			sgd(X,Y)
			perceptron(X,Y)
			clustering(X,Y)
			

	
	rdd=word.foreachRDD(main)
	print("new batch")
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()






