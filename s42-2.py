import pandas as pd
data=pd.read_excel("stu_reading&browsingTimes.xlsx")
df=pd.DataFrame(data)

x=df[["study percentage","browsing times"]]
y=df["test result"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
df2=pd.DataFrame({"ytest":ytest,"ypred":ypred})
print(df2)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))

new_sample = pd.DataFrame([[40, 2]], columns=["study percentage", "browsing times"])
print("predict=", model.predict(new_sample))

