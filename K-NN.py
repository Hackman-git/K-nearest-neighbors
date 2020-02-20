'''
Name:   Olugbenga Abdulai
ID:     A20447331
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as lin
from sklearn.neighbors import KNeighborsClassifier as KNN


# Reading the file
path = r"C:\Users\abdul\Desktop\CS 584\HW\HW 1\Fraud.csv"
fraud_data = pd.read_csv(path)

'''
Q 3(a): percentage of fraudulent cases
'''
# obtaining the fraud cases from the dataframe
fraud_cases = fraud_data.loc[fraud_data.FRAUD == 1, :]
fraud_cases_count = fraud_cases.CASE_ID.count()

# calculating percent of fraudulent cases
total_count = fraud_data.CASE_ID.count()
percent_fraud = fraud_cases_count / total_count * 100
print('percentage of fraudulent cases: ', percent_fraud)

'''
Q 3(b): Boxplots for each interval variable
'''
sns.boxplot(x=fraud_data.TOTAL_SPEND, y=fraud_data.FRAUD, orient='horizontal')
plt.show()

sns.boxplot(x=fraud_data.DOCTOR_VISITS, y=fraud_data.FRAUD, orient='horizontal')
plt.show()

sns.boxplot(x=fraud_data.NUM_CLAIMS, y=fraud_data.FRAUD, orient='horizontal')
plt.show()

sns.boxplot(x=fraud_data.MEMBER_DURATION, y=fraud_data.FRAUD, orient='horizontal')
plt.show()

sns.boxplot(x=fraud_data.OPTOM_PRESC, y=fraud_data.FRAUD, orient='horizontal')
plt.show()

sns.boxplot(x=fraud_data.NUM_MEMBERS, y=fraud_data.FRAUD, orient='horizontal')
plt.show()

'''
Q 3(c): Transforming the predictor matrix
'''
# getting the predictors
predictors = fraud_data.loc[:, ['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION',
                                'OPTOM_PRESC', 'NUM_MEMBERS']]
pred_matrix = np.array(predictors)
print('predictor matrix:\n', pred_matrix)

# getting the number of dimensions
print("number of dimensions: ", pred_matrix.ndim)

x_transp_x = np.matmul(pred_matrix.transpose(), pred_matrix)
print('x_transpose * x:\n', x_transp_x)

# getting the eigenvalues and eigenvectors
eigen_vals, eigen_vecs = lin.eigh(x_transp_x)
print('eigenvalues: \n', eigen_vals)
print('eigenvectors:\n', eigen_vecs)

# obtaining the transformation matrix
diag_matrix = lin.inv(np.sqrt(np.diagflat(eigen_vals)))
transf_matrix = np.matmul(eigen_vecs, diag_matrix)
print('transformation matrix:\n', transf_matrix)

# obtaining the transformed predictor matrix
transf_pred = np.matmul(pred_matrix, transf_matrix)
print('transformed predictor matrix:\n', transf_pred)

# To prove that the transformed predictor matrix is orthonormal
# we multiply it by its transpose and should expect an identity matrix
ident = np.matmul(transf_pred.transpose(), transf_pred)
print('The identity matrix is:\n', ident)

'''
Q 3(d): score of the nearest neighbor algorithm
'''
knn = KNN(n_neighbors=5)
nbrs = knn.fit(X=transf_pred, y=fraud_data.FRAUD)
score = knn.score(X=transf_pred, y=fraud_data.FRAUD)
print('KNN analysis score: ', score)

'''
Q 3(e): Finding 5 nearest neighbors for a sample observation
'''
# sample input values
test_x = np.array([[7500, 15, 3, 127, 2, 2]])

# transforming the input
transf_test_x = np.matmul(test_x, transf_matrix)
print('transformed test x:\n', transf_test_x)

# obtaining the neighbors
test_x_neighbors = knn.kneighbors(transf_test_x, n_neighbors=5, return_distance=False)
print('neigbors of test input:\n', test_x_neighbors)

# obtaining and printing the input values and labels of the neighbors
results = test_x_neighbors[0]
for i in results:
    print('case '+str(i) +' inputs:', fraud_data.loc[i, ['TOTAL_SPEND', 'DOCTOR_VISITS',
                'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']], end='\n\n')
    print('case '+str(i) +' label:', fraud_data.loc[i, ['FRAUD']], end='\n\n')

'''
Q 3(f): predicted probability of fraud
'''
# obtaining the probability value of fraud
prob = nbrs.predict_proba(transf_test_x)
target_class = ['not fraud', 'fraud']
print('probability values for the test input: ', prob)

best_prob_index = np.argmax(prob[0])
print('According to the criterion (the probability '
      'is >0.2), the test input is predicted as: '+ target_class[best_prob_index])

