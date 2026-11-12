import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
df = pd.read_csv(r"D:/datasets/creditcard.csv")

print(df.shape)
print(df.isnull().sum())
Class_counts = df["Class"].value_counts()
Class_percents = df["Class"].value_counts(normalize=True)*100
print(f"{Class_counts[0]},{Class_percents[0]:.2f}")
print(f"{Class_counts[1]},{Class_percents[1]:.2f}")
print(df[df["Class"]==1]["Amount"].describe())
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sb.countplot(x="Class",data=df,palette="Set1")
plt.title("Class Distribution (Log Scale)")
plt.yscale("log")
plt.subplot(1,2,2)
sb.boxplot(x=("Class"),y="Amount",data =df,palette="Set1")
plt.title("Amount Distribution Per Class")
plt.ylim(0,2000)
plt.tight_layout()
plt.show()
scalar = RobustScaler()
df["Scaled Amount"] = scalar.fit_transform(df["Amount"].values.reshape(-1,1))
df["Scaled Time"] = scalar.fit_transform(df["Time"].values.reshape(-1,1))
df.drop(["Time","Amount"],axis=1,inplace=True)
x = df.drop("Class",axis=1)
y = (df["Class"])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print (x_test.shape)
print(x_test.shape)
print(f"{y_train.mean()*100:.3f}%")
print(f"{y_test.mean()*100:.3f}%")
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,precision_recall_curve,auc
smote = SMOTE(random_state=42)
x_train_res,y_train_res = smote.fit_resample(x_train,y_train) 
print(f"number of samples before: {y_train.value_counts().to_dict()}")
print(f"number of samples after: {y_train_res.value_counts().to_dict()}")
model = XGBClassifier(
    n_estimators = 200,
    max_depth = 7,
    learning_rate = 0.1,
    random_state = 42
)
model.fit(x_train_res,y_train_res)  
y_prob = model.predict_proba(x_test)[:,1]
custom_threshold = 0.85 
y_pred = (y_prob >= custom_threshold).astype(int)
print(classification_report(y_test,y_pred))
precition,recall,_=precision_recall_curve(y_test,y_prob)
pr_auc = auc(recall,precition)
plt.figure(figsize=(7,5))
plt.plot(recall,precition,label = f"XGBoost (PR-AUC = {pr_auc:.4f})",color = "blue")
plt.xlabel("Recall")
plt.ylabel("Precition")
plt.title("Precition-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

importances = model.feature_importances_
feature_names = x.columns
indices = np.argsort(importances)[::-1]
top_indices = indices[:10]
plt.figure(figsize = (10,6))
plt.title("top 10 features model depended on ",fontsize =14)
plt.barh(range(10),importances[top_indices][::-1],align="center",color = "skyblue")
plt.yticks(range(10), [feature_names[i] for i in top_indices][::-1])
plt.xlabel("(Feature Importance Score)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
