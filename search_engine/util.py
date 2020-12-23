from collections import defaultdict
import collections
import math
import numpy as np
from array import array
import datetime
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from numpy import linalg as la

def getTerms(line):        
	stemming = PorterStemmer()
	stops = set(stopwords.words("english"))

	line=  line.lower()
	pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	line = pattern.sub('', line)
	line = re.sub('[^a-zA-Z]', ' ', line )                                
	line=  line.split()                                 
	line=[word for word in line if word not in stops]   
	line=[stemming.stem(word) for word in line]        

	return line

def create_index_tfidf_tp(df_tweets):
	numDocuments = len(df_tweets)
	index = defaultdict(list)
	tf = defaultdict(
			list
	)  # term frequencies of terms in documents (documents in the same order as in the main index)
	df = defaultdict(int)  # document frequencies of terms in the corpus
	idf = defaultdict(float)
	tp = defaultdict(int)
	itp = defaultdict(float)

	current_timestamp = datetime.datetime.now().timestamp()
	for i, row in df_tweets.iterrows():
			terms = row["processed_text"]
			user_created_at = row["user"]["created_at"]
			user_followers = row["user"]["followers_count"]

			user_creation_date = datetime.datetime.strptime(
					user_created_at, "%a %b %d %H:%M:%S +%f %Y"
			)
			user_creation_timestamp = user_creation_date.timestamp()

			page_id = i
			termdictPage = {}

			for position, term in enumerate(
					terms
			):  ## terms contains page_title + page_text
					try:
							# if the term is already in the dict append the position to the corrisponding list
							termdictPage[term][1].append(position)
					except:
							# Add the new term as dict key and initialize the array of positions and add the position
							termdictPage[term] = [
									page_id,
									array("I", [position]),
							]  #'I' indicates unsigned int (int in python)

			norm = 0
			for term, posting in termdictPage.items():
					norm += len(posting[1]) ** 2
			norm = math.sqrt(norm)

			# calculate the tf(dividing the term frequency by the above computed norm) and df weights
			for term, posting in termdictPage.items():

					# append the tf for current term (tf = term frequency in current doc/norm)
					tf[term].append(
							np.round(len(posting[1]) / norm, 4)
					)  ## SEE formula (1) above
					# increment the document frequency of current term (number of documents containing the current term)
					df[term] = df[term] + 1  # increment df for current term

			for termpage, postingpage in termdictPage.items():
					tp[termpage] += np.round(
							(np.log(current_timestamp - user_creation_timestamp) * user_followers),
							4,
					)
					index[termpage].append(postingpage)

			# Compute idf following the formula (3) above. HINT: use np.log
	for term in df:
			idf[term] = np.round(np.log(float(numDocuments / df[term])), 4)
			if(tp[term] != 0):
				itp[term] = np.round(np.log(float(numDocuments / tp[term])), 4)
			itp[term] = 0

	return index, tf, df, idf, tp, itp


def rankDocuments_tf_idf(terms, docs, index, idf, tf):
	docVectors = defaultdict(lambda: [0] * len(terms))
	queryVector = [0] * len(terms)

	# compute the norm for the query tf
	query_terms_count = collections.Counter(
			terms
	)  # get the frequency of each term in the query.

	query_norm = la.norm(list(query_terms_count.values()))

	for termIndex, term in enumerate(
			terms
	):  # termIndex is the index of the term in the query
			if term not in index:
					continue
			queryVector[termIndex] = query_terms_count[term] / query_norm * idf[term]

			# Generate docVectors for matching docs
			for docIndex, (doc, postings) in enumerate(index[term]):

					if doc in docs:
							docVectors[doc][termIndex] = (
									tf[term][docIndex] * idf[term]
							)  # TODO: check if multiply for idf

	docScores = [
			[np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items()
	]
	docScores.sort(reverse=True)
	resultDocs = [x[1] for x in docScores]

	return resultDocs


def search_tf_idf(query, index, tf, idf):
	"""
	output is the list of documents that contain any of the query terms.
	So, we will get the list of documents for each query term, and take the union of them.
	"""
	query = getTerms(query)
	docs = set()
	for term in query:
			try:
					# store in termDocs the ids of the docs that contain "term"
					termDocs = [posting[0] for posting in index[term]]

					# docs = docs Union termDocs
					docs |= set(termDocs)
			except:
					# term is not in index
					pass
	docs = list(docs)
	ranked_docs = rankDocuments_tf_idf(query, docs, index, idf, tf)
	return ranked_docs

def rankDocuments_itp(terms, docs, index, tf, itp):
	"""
	Returns the list of ranked docs
	"""     

	docVectors=defaultdict(lambda: [0]*len(terms)) 	
	queryVector=[0]*len(terms)    

	# compute the norm for the query tf
	query_terms_count = collections.Counter(terms) 
	
	query_norm = la.norm(list(query_terms_count.values()))
	
	for termIndex, term in enumerate(terms): #termIndex is the index of the term in the query
			if term not in index:
					continue         
			## Compute tf*idf(normalize tf as done with documents)
			queryVector[termIndex]=query_terms_count[term]/query_norm * itp[term] 
			# Generate docVectors for matching docs
			for docIndex, (doc, postings) in enumerate(index[term]):
       
					if doc in docs:
							docVectors[doc][termIndex]=tf[term][docIndex] * itp[term]  
	
	docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]
	docScores.sort(reverse=True)
	resultDocs=[x[1] for x in docScores]

	return resultDocs

def search_itp(query, index, tf,itp):
	'''
	output is the list of documents that contain any of the query terms. 
	So, we will get the list of documents for each query term, and take the union of them.
	'''
	query=getTerms(query)
	docs=set()
	for term in query:
		try:
				# store in termDocs the ids of the docs that contain "term"                        
				termDocs=[posting[0] for posting in index[term]]
				
				# docs = docs Union termDocs
				docs|=set(termDocs)
		except:
				#term is not in index
				pass
	docs=list(docs)
	ranked_docs = rankDocuments_itp(query, docs, index, tf, itp)   
	return ranked_docs