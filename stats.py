#statistical bull
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

	#was part of doAll()
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