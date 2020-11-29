import asyncio
import numpy as np
import pandas as pd

# returns x and y sets based on desired prediction column
def getSets(df, col_predict):
    df = df.dropna().reset_index(drop=True)
    cols = df.shape[1]
    dropCol = list(df.columns.values)[col_predict]
    x = df.drop(columns=dropCol).iloc[:, 0:cols].values
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    y = df.iloc[:, col_predict].values
    return df, x, y

# implementation of logistic regression functions
class LogisticRegression:
    def __init__(self):
        self.lr = 0.05
        self.iterations = 5000

    def sigmoid(self, z):
        z = z.astype('float128')
        return 1 / (1 + np.exp(-z))

    def gradient(self, x, y, theta):
        h = self.sigmoid(np.dot(x, theta))
        return np.dot(x.T, (h-y)) / len(y)

    async def fit(self, x, y):
        theta = np.zeros(x.shape[1])
        for i in range(self.iterations):
            theta -= self.lr * self.gradient(x, y, theta)
        return theta

    async def predict(self, x, theta, thresh):
        return (self.sigmoid(np.dot(x, theta)) >= thresh)*1

    async def trainAndPredict(self, x, y, thresh):
        theta = await self.fit(x, y)
        preds = await self.predict(x, theta, thresh)
        accuracy = sum(preds == y) / len(y)
        return accuracy

# implementation of ROC curve functions
class ROC_Curve:
    def getTPR(self, conf):
        if (conf['tp'] > 0 or conf['fn'] > 0):
            return conf['tp'] / (conf['tp'] + conf['fn'])
        else:
            return conf['tp'] / 1

    def getFPR(self, conf):
        if (conf['tn'] > 0 or conf['fp'] > 0):
            return conf['fp'] / (conf['tn'] + conf['fp'])
        else:
            return conf['fp'] / 1

    def confusionMatrix(self, y, preds):
        TP = FP = TN = FN = 0
        for i in range(len(y)):
            if (y[i]):
                if (preds[i]):
                    TP += 1
                else:
                    FN += 1
            else:
                if (preds[i]):
                    FP += 1
                else:
                    TN += 1
        return {'tp': TP, 'fp': FP, 'tn': TN, 'fn': FN}

    async def getCurve(self, x, y):
        reg = LogisticRegression()
        theta = await reg.fit(x, y)
        thresholds = np.arange(0, 1, 0.001)
        tpr = []
        fpr = []
        matrices = []
        for thresh in thresholds:
            preds = await reg.predict(x, theta, thresh)
            conf = self.confusionMatrix(y, preds)
            matrices.append(conf)
            tpr.append(self.getTPR(conf))
            fpr.append(self.getFPR(conf))
        return tpr, fpr, matrices