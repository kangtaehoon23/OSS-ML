from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

print(cancer)

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1234)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("정확율:", accuracy)