#!/usr/bin/env python
# coding: utf-8

# # Q1. What is the mathematical formula for a linear SVM?

# a linear Support Vector Machine (SVM) is a machine learning model used for classification. The goal of this model is to find a straight line (or hyperplane) that best separates different classes in the data.
# 
# The prediction for a new input is made by looking at which side of the line the input falls on. If it's on one side, it's classified as one group, and if it's on the other side, it's classified as another group.
# 
# w \cdot x + b = 0
# 
# w is a weight vector that represents the direction of the hyperplane
# 
# x is an input vector
# 
# b is a bias term that represents the offset of the hyperplane from the origin
# 
# 

# # Q2. What is the objective function of a linear SVM?

# The objective function of a linear Support Vector Machine (SVM) is designed to find the optimal hyperplane that maximally separates different classes in the training data. The objective function is typically expressed as the minimization of a certain mathematical expression.
# 
# For a linear SVM, the objective function involves two components:
# 
# 1. **Maximizing the Margin:**
#    The SVM aims to maximize the margin between the two classes. The margin is the distance between the hyperplane and the nearest data point from either class. Mathematically, this is achieved by maximizing the reciprocal of the norm (length) of the weight vector.
# 
#    \[ \text{maximize} \left( \frac{1}{\|\mathbf{w}\|} \right) \]
# 
#    Here, \(\mathbf{w}\) represents the weight vector of the hyperplane.
# 
# 2. **Correctly Classifying Data:**
#    The SVM also includes a term that ensures the correct classification of training data. It introduces a penalty for misclassified points, encouraging the model to correctly classify each training example.
# 
#    \[ \text{subject to } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \text{ for all } i \]
# 
#    Here, \(y_i\) is the class label of the ith training example, \(\mathbf{x}_i\) is the feature vector, and \(b\) is the bias term.
# 
# The overall objective function for a linear SVM is to find the values of \(\mathbf{w}\) and \(b\) that simultaneously maximize the margin and minimize the misclassification error. This is typically formulated as a constrained optimization problem and is solved using techniques such as quadratic programming.
# 

# # Q3. What is the kernel trick in SVM?

# The kernel trick is a technique that enables support vector machines (SVMs) to effectively handle non-linear relationships between data points. SVM is a powerful classification algorithm that can effectively separate data points into two classes by finding a hyperplane that maximizes the margin between the two classes. However, when dealing with non-linear data, a linear hyperplane may not be sufficient to accurately separate the data points. This is where the kernel trick comes into play.
# 
# The kernel trick essentially maps the data points from the original low-dimensional space into a higher-dimensional feature space. In this higher-dimensional space, the relationships between the data points may become more linear, allowing the SVM to find a hyperplane that effectively separates the data points. However, instead of explicitly performing the mapping to the higher-dimensional space, the kernel trick utilizes a kernel function to compute the dot product between data points in the original space. This allows the SVM to perform the computations as if it were operating in the higher-dimensional space without explicitly transforming the data points.
# 
# Commonly used kernel functions include:
# 
# **Linear kernel:** This kernel is used for linear SVMs and simply computes the dot product between data points in the original space.
# 
# **Polynomial kernel:** This kernel raises the dot product of data points to a certain power, which allows for more complex relationships to be captured.
# 
# **Gaussian kernel:** This kernel computes the similarity between data points using a Gaussian function, which is a bell-shaped curve. This kernel is effective for capturing non-linear relationships between data points that are close together.
# 
# The kernel trick is a powerful tool that allows SVMs to handle non-linear relationships between data points without explicitly transforming the data into a higher-dimensional space. This makes SVMs a versatile and effective classification algorithm for a wide range of problems.

# # Q4. What is the role of support vectors in SVM Explain with example
# 

# Support vectors play a crucial role in support vector machines (SVMs), a powerful machine learning algorithm for classification and regression tasks. They are the data points that define the decision boundary, the hyperplane that separates data points into two or more classes. These data points are the most influential in determining the position and orientation of the hyperplane, making them critical for the overall performance of the SVM.
# 
# Consider an example of classifying fruits based on their color and size. Imagine a scatter plot where each fruit is represented as a data point, with color represented by the x-axis and size represented by the y-axis. The goal of the SVM is to find a hyperplane that separates the fruits into two classes, say apples and oranges.
# 
# During the training process, the SVM algorithm identifies the support vectors, which are the fruits that lie closest to the hyperplane. These support vectors essentially define the boundaries of the decision-making region. As new fruits are introduced, the SVM classifies them based on their position relative to the support vectors and the hyperplane.
# 
# The role of support vectors in SVM can be summarized as follows:
# 
# **Defining the decision boundary:** Support vectors determine the position and orientation of the hyperplane, which separates the data points into different classes.
# 
# **Maximizing the margin:** Support vectors are the data points that lie closest to the hyperplane, contributing to the maximum margin. The margin is the distance between the hyperplane and the closest data points from both classes.
# 
# **Reducing sensitivity to outliers:** Support vectors are less affected by outliers or noisy data points, making the SVM more robust and less prone to overfitting.
# 
# **Improving generalization performance:** By focusing on the most influential data points, support vectors contribute to the SVM's ability to generalize well to new, unseen data.

# # Q5. Illustrate with examples and graphs of Hyperplane, Marginal plane, Soft margin and Hard margin in SVM?

# 1. Hyperplane:
# 
# In a Support Vector Machine (SVM), the hyperplane is a decision boundary that separates different classes in a feature space. For a binary classification problem in a two-dimensional feature space, the hyperplane is a line.
# 
# 2. Marginal Plane:
# 
# The marginal plane consists of parallel lines to the hyperplane that pass through the support vectors (data points closest to the hyperplane).The margin is the distance between the hyperplane and the marginal planes. The wider the margin, the better the SVM generalizes to new, unseen data. The support vectors determine the position and orientation of the marginal planes.
# 
# 3. Hard Margin:
# 
# Hard margin SVM aims to find a hyperplane that perfectly separates different classes without allowing any misclassifications.
# This approach is suitable when the data is linearly separable, and there is no tolerance for errors. However, it may be impractical or lead to overfitting when dealing with noisy or overlapping data.
# 
# 4. Soft Margin:
# It refers to a flexible approach that allows for some degree of misclassification in the training data to achieve a wider margin between classes. Soft margin SVM is particularly useful when dealing with data that is not perfectly separable or contains noise.
# 
# ![svm%20assign.png](attachment:svm%20assign.png)

# In[ ]:





# # Q6. SVM Implementation through Iris dataset.
# 
# ### ~ Load the iris dataset from the scikit-learn library and split it into a training set and a testing setl
# ### ~ Train a linear SVM classifier on the training set and predict the labels for the testing setl
# ### ~ Compute the accuracy of the model on the testing set
# ### ~ Plot the decision boundaries of the trained model using two of the featuresl
# ### ~ Try different values of the regularisation parameter C and see how it affects the performance of the model.

# In[14]:


from sklearn.datasets import load_iris
dataset = load_iris()


# In[15]:


print(dataset.DESCR)


# In[18]:


dataset.keys()


# In[19]:


import pandas as pd
X = X =pd.DataFrame(dataset.data, columns=dataset.feature_names)
X.head()


# In[20]:


dataset.target_names


# In[21]:


Y = pd.DataFrame(dataset.target, columns=['class'])
Y.head()


# In[22]:


from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain, ytest =  train_test_split(X,Y,test_size=0.33,random_state=42)


# In[23]:


xtrain.shape


# In[24]:


xtest.shape


# In[30]:


from sklearn.svm import SVC
svm= SVC(kernel='linear',C=1.0)
svm.fit(xtrain, ytrain.values.flatten())


# In[32]:


y_pred = svm.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ytest, y_pred)
print("Testing Accuracy:", accuracy)


# In[33]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cf = confusion_matrix(ytest,y_pred)
sns.heatmap(cf,annot=True)


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt
df_train = pd.concat([xtrain,ytrain],axis=1)
sns.scatterplot(data = df_train, x = 'petal length (cm)',y='petal width (cm)',hue='class')
plt.title('Petal length vs Petal width hueplot')
plt.show()


# In[ ]:




