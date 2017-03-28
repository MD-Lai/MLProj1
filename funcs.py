'''
Functions written for Comp30027 - Machine Learning Project 1
Specifically written to process the abalone dataset
Requires random library to work 
Written for Python 3.4
'''
import random
#takes raw data and returns it as a tuple consisting of 
#([attributes], [[data1],[data2],...,[datan]])
def preprocess_data(dir):

	f = open(dir, 'r')
	attributes = ["Sex",
	"Length",
	"Diameter",
	"Height",
	"Whole Weight",
	"Shucked Weight",
	"Viscera Weight",
	"Shell weight",
	"Rings"]
	data = (attributes,[])

	
	for line in f:
		thisLine = line.rstrip().split(',')
		
		thisLine[0]  = 0 if thisLine[0] == 'I' else 1
		
		for i in range(1,len(thisLine) - 1):
			thisLine[i] = float(thisLine[i])

		#bin at start or bin at end...that is the question
		#i say bin at start
		thisLine[len(thisLine) - 1] = bin(float(thisLine[len(thisLine) - 1]), 1)
		#thisLine[len(thisLine) - 1] = float(thisLine[len(thisLine) - 1])
		data[1].append(thisLine)
	f.close()
	return data

def bin(rings, abMode = 2):
	if(abMode == 1):
		return rings

	elif(abMode == 2):
		return 0 if rings <= 10 else 1 

	elif(abMode == 3):
		if(rings <= 8):
			return 0
		elif(rings <= 9 and rings <= 10):
			return 1
		else:
			return 2

	return -1
#compares a test instance to a training instance
#using method (default euclidean dist) to calculate similarity (lower is closer)
#with optional attibutes to exclude in comparisons specified as array of strings
def compareInstance(testInst, trainingInst, method = "euc", excl = []):
	'''
	Methods available:
		"euc" - euclidean distance (disregarding sex)
		"mnh" - manhattan distance (disregarding sex)
		"cos" - cosine similarity (disregarding sex)

	Exclusions are optional strings inputted as an array 
	to exclude particular attributes
	'''
	total = 0
	attributes = ["Sex",
	"Length",
	"Diameter",
	"Height",
	"Whole Weight",
	"Shucked Weight",
	"Viscera Weight",
	"Shell weight",
	"Rings"] # to exclude rings in comparison, add "Rings" to excl

	if(method == "euc"):
		#how to account for M/F/I...
		for i in range(len(testInst)):
			if attributes[i] not in excl:
				total += pow(trainingInst[i] - testInst[i], 2)

		return pow(total, 0.5)

	if(method == "mnh"):
		for i in range(len(testInst)):
			if attributes[i] not in excl:
				total += abs(trainingInst[i] - testInst[i])
		return total

	# Note this is Cosine Difference (1-sim)
	# to follow other return values where smaller values == more similar
	# Cosine similarity may be used but is basically useless
	# since things tend to increase in all dimensions as they grow
	if(method == "cos"):
		pq = 0
		p = 0
		q = 0
		for i in range(len(testInst)):
			if attributes[i] not in excl:
				pq += trainingInst[i] * testInst[i]
				p += pow(trainingInst[i], 2)
				q += pow(testInst[i], 2)

		return 1 - pq/(pow(p,0.5) * pow(q,0.5))

	return False

#gets the k closest neighbours (default 3)
#using the specified method (default euclidean dist) to calculate similarity
def getNeighbours(testInst, trainingDataSet, k = 3, method = 'euc', excl= []):
	# (class, score), looking to collect lowest scores
	top = []
	if(k > len(trainingDataSet[1])):
		k = len(trainingDataSet[1])

	for i in range(k):
		top.append((0, 9999))

	for trainInst in trainingDataSet[1]:
		diff = compareInstance(testInst, trainInst, method, excl)
		i = len(top) - 1
		#large diff means more different, small diff mean less different
		#meaning we want small diff at back to be reversed later
		while(i >= 0 and diff > top[i][1]):
			i -= 1
		if(i >= 0):
			top.insert(i + 1, (trainInst[len(trainInst) - 1], diff))
			top.remove(top[i])
	# reverses the list of tuples, 
	# avoids having to see "len(trainInst) - 1" everywhere instead of "0"
	return list(reversed(top))

#used to verify that the given list of closest neighbours is sequential
def verifyTop(topList):
	for i in range(len(topList) - 1):
		if(topList[i][1] < topList[i+1][1]):
			return False

	return topList

#predicts a class using specified method (default knn)
def predictClass(neighbours, method = 'knn'):
	'''
	Methods available:
	"1nn" - classifies instance based on single nearest neighbour
	"knn" - classifies instance based on majority class of k nearest neighbours
	'''
	if(method == "1nn"):
		return neighbours[0][0]

	if(method == "knn"):
		cls = 0
		tracker = {}
		for clsScore in neighbours:
			if(clsScore[0] in tracker):
				tracker[clsScore[0]] += 1#.append(clsScore)
			else:
				tracker[clsScore[0]] = 1#[clsScore]

		items = []
		for trackerKeys in tracker.keys():
			items.append((trackerKeys, tracker[trackerKeys]))


		for i in range(1, len(items) - 1):
			j = i 

			# want item with highest number of close neighbours to be at front
			# swap if item below it has lower number of neighbours
			#while(j > 0 and len(items[j-1][1]) < len(items[j][1])):
			while(j > 0 and items[j-1][1] < items[j][1]):
				items[j-1], items[j] = items[j], items[j-1]
				j -= 1

		return items[0][0]


	return False

#evaluate performance of class prediction 
#using p fold cross validation (default 5)
#please give p <= number of data points or it will throw an error
#optionally takes number of neighbours for classifier(default 5)
def evaluate(data_set, metric = "accuracy", p = 5, nn = 5):
	#data_set is ([attributes], [[data1],[data2],...,[datan]])

	attributes = data_set[0]
	tp = 0
	fn = 0
	tn = 0
	fp = 0
	random.shuffle(data_set[1])
	#k fold cross verification
	for a in range(p):

		start = int((len(data_set[1]) / p) * a) 
		end = min(start + int(len(data_set[1]) / p), int(len(data_set[1])))
		samples = list(range(start, end))


		testSet = []
		trainSet = [attributes, []]
		resultSet = [] #(classifier class, actual class)

		for i in range(len(data_set[1]) - 1):
			if i in samples:
				testSet.append(data_set[1][i])
			else:
				trainSet[1].append(data_set[1][i])
		
		print("Predicting classes...")
		#lord almighty this is slow...
		for testInstance in testSet:

			resultSet.append(
				
				(bin(
					predictClass(
						getNeighbours(
							testInstance,trainSet, nn, "cos", ["Rings"]), "knn")
					, 2), 
				bin(testInstance[attributes.index("Rings")] ,2))
			)

		print("Set", a+1, "of", p, "complete")
		printPredictions(resultSet)
		#print("Results -", len(resultSet), "predictions", resultSet)
		#now you have (classified class, actual class) now calculate the metric
		#but how do you determine tn or fp
		#accuracy = tp+tn/nGuesses in this case just tp/tp+fp

		
		for result in resultSet:
			if(result[0] == result[1] and result[0] == 1):
				tp += 1
			elif(result[0] == result[1] and result[0] == 0):
				tn += 1

			elif(result[0] != result[1] and result[0] == 1):
				fp += 1
			elif(result[0] != result[1] and result[0] == 0):
				fn += 1

	accuracy = (tp + tn)/(tp+fp+fn+tn)
	error_rate = 1 - accuracy
	precision = tp/(tp+fp)
	sensitivity = tp/(tp+fn)
	specificity = tn/(tn+fp)

	'''
	print("acc:", accuracy)
	print("err:", error_rate)
	print("pre:", precision)
	print("sen:", sensitivity)
	print("spe:", specificity)
	'''

	if(metric == "accuracy"):
		return accuracy
	if(metric == "error_rate"):
		return error_rate
	if(metric == "precision"):
		return precision
	if(metric == "sensitivity"):
		return sensitivity
	if(metric == "specificity"):
		return specificity
	if(metric == "all"):
		return [accuracy, error_rate, precision, sensitivity, specificity]


def printPredictions(predicts):
	nYoung = 0
	nOld = 0
	cYoung = 0
	cOld = 0

	for guess in predicts:

		if(guess[1] == 0):
			nYoung += 1
		elif(guess[1] == 1):
			nOld += 1

		if(guess[0] == guess[1]):
			if(guess[0] == 0):
				cYoung += 1
			elif(guess[0] == 1):
				cOld += 1
	
	print("n Yng", nYoung, "c Yng", cYoung)
	print("n Old", nOld, "c Old", cOld)

#some statistical analysis stuff regarding the raw data
def avgRings(data, sex):

	total = 0
	totalDist = 0
	totalSkew = 0
	nInstances = 0
	for instv1 in data[1]:
		if instv1[0] == sex:
			total += instv1[len(instv1) - 1]
			nInstances += 1

	mean = total/nInstances

	for instv2 in data[1]:
		if instv2[0] == sex:
			totalDist += pow(instv2[len(instv2) - 1] - mean, 2)/nInstances

	# standard deviation measures distribution of data
	# only +ve numbers
	sd = pow(totalDist,0.5)

	print('sd', sex, sd)

	for instv3 in data[1]:
		if instv3[0] == sex:
			totalSkew += pow((instv3[len(instv3) - 1] - mean)/sd, 3)/nInstances

	# skewness how skewed the data is, 
	# +ve means more outliers in the high range
	# -ve means more outliers in the low range
	print('sq', sex, totalSkew)

	return mean

def printData(data, sampleFreq = 100):
	print(data[0])

	for i in range(len(data[1])):
		if i%sampleFreq == 0 or i == len(data[1]) - 1: 
			print(i, data[1][i])

def printStats(data):
	print("av I", avgRings(data, "I"), '\n')
	print("av M", avgRings(data, "M"), '\n')
	print("av F", avgRings(data, "F"), '\n')

def doAll():

	#procData = preprocess_data('./abalone/abalone.data')

	#print(procData)

	#printStats(procData)


	#indexToRemove = random.randrange(0, len(procData[1]))
	#testCase = procData[1][indexToRemove]

	#procData[1].remove(procData[1][indexToRemove])

	'''
	print(compareInstance(
		# random.randrange(0, len(procData[1]))
		testCase, 
		procData[1][random.randrange(0, len(procData[1]))],
		"euc",
		["Rings"])
		)
	'''

	#neighbours = getNeighbours(testCase, procData, k = 40)
	#print(neighbours)
	#print(predictClass(neighbours, "knn"))
	#print(testCase[procData[0].index("Rings")])
	acc = []
	folds = [19]
	nei = [11,15,17,20]
	for p in folds:
		for nn in nei:
			acc.append(
				[p,nn, evaluate(
					preprocess_data('./abalone/abalone.data'), "all", p, nn)]
				)
	f = open("results.txt", "a")
	
	for a in acc:
		results = ""
		results += "Splits: " +str( a[0])+ " nn "+str(a[1]) + "\n"
		results += "acc: " + str(a[2][0]) + "\n"
		results += "err: " + str(a[2][1]) + "\n"
		results += "pre: " + str(a[2][2]) + "\n"
		results += "sen: " + str(a[2][3]) + "\n"
		results += "spe: " + str(a[2][4]) + "\n"
		results += "\n"

		f.write(results)
		'''
		print("Splits", a[0], "nn", a[1])
		print("acc:", a[2][0]) #percentage of times it guessed right
		print("err:", a[2][1]) #percentage of times it guessed wrong
		print("pre:", a[2][2]) #percent of times it correctly guessed old
		print("sen:", a[2][3]) #sensitivity is n old
		print("spe:", a[2][4]) #specificity is n young
		'''
	f.close()
def testPrint():
	f = open("test.txt",'a')
	a = 5
	string = "testing" + str(a) + "\n"
	for i in range(5):
		f.write(str(i) +" "+ string)
	f.close()
doAll()
#testPrint()