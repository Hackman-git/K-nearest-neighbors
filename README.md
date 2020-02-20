# K-nearest-neighbors
The data, FRAUD.csv, contains results of fraud investigations of 5,960 cases.  The binary variable FRAUD indicates the result of a fraud investigation: 1 = Fraudulent, 0 = Otherwise.  The other interval variables contain information about the cases.
1.	TOTAL_SPEND: Total amount of claims in dollars
2.	DOCTOR_VISITS: Number of visits to a doctor  
3.	NUM_CLAIMS: Number of claims made recently
4.	MEMBER_DURATION: Membership duration in number of months
5.	OPTOM_PRESC: Number of optical examinations
6.	NUM_MEMBERS: Number of members covered
You are asked to use the Nearest Neighbors algorithm to predict the likelihood of fraud.
a)	(5 points) What percent of investigations are found to be fraudulent?  Please give your answer up to 4 decimal places.

b)	(5 points) Use the BOXPLOT function to produce horizontal box-plots.  For each interval variable, one box-plot for the fraudulent observations, and another box-plot for the non-fraudulent observations.  These two box-plots must appear in the same graph for each interval variable.

c)	(10 points) Orthonormalize interval variables and use the resulting variables for the nearest neighbor analysis. Use only the dimensions whose corresponding eigenvalues are greater than one.
i.	(5 points) How many dimensions are used?
ii.	(5 points) Please provide the transformation matrix?  You must provide proof that the resulting variables are actually orthonormal.

d)	(10 points) Use the NearestNeighbors module to execute the Nearest Neighbors algorithm using exactly five neighbors and the resulting variables you have chosen in c).  The KNeighborsClassifier module has a score function.

i.	(5 points) Run the score function, provide the function return value
ii.	(5 points) Explain the meaning of the score function return value.

e)	(5 points) For the observation which has these input variable values: TOTAL_SPEND = 7500, DOCTOR_VISITS = 15, NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2, find its five neighbors.  Please list their input variable values and the target values. Reminder: transform the input observation using the results in c) before finding the neighbors.

f)	(5 points) Follow-up with e), what is the predicted probability of fraudulent (i.e., FRAUD = 1)?  If your predicted probability is greater than or equal to your answer in a), then the observation will be classified as fraudulent.  Otherwise, non-fraudulent.  Based on this criterion, will this observation be misclassified?
