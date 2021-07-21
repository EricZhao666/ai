import pandas as pd
df_housing = pd.read_csv("house.csv")
df_housing.head
X = df_housing.drop("median_house_value",axis = 1)# 去掉最后的中位数作为特征集
y = df_housing.median_house_value# 将中位数作为标签集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)# 拆分进行训练和预测
from sklearn.linear_model import LinearRegression
model = LinearRegression()# 运用线性回归模型预测
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('实际',y_test)
print('预测',y_pred)
print('评分',model.score(X_test,y_test))
