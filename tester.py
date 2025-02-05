import os
from skimage.io import imread
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage.transform import resize
import numpy as np
dir='D:\\project\\clf-data\\clf-data'
categories=['empty','not_empty']
data=[]
label=[]
for categorind,category in enumerate(categories):
    for file in os.listdir(os.path.join(dir,category)):
        img_path=os.path.join(dir,category,file)
        img=imread(img_path)
        img=resize(img,(15,15))
        data.append(img.flatten())
        label.append(categorind)
data=np.asarray(data)
label=np.asarray(label)
#train test split
train_x,test_x,train_y,test_y=train_test_split(data,label,test_size=0.2,shuffle=True,stratify=label)
classifier =SVC()
parameters=[{'gamma':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]
grid_search=GridSearchCV(classifier,parameters)
grid_search.fit(train_x,train_y)
best=grid_search.best_estimator_
y_pred=best.predict(test_x)
score=accuracy_score(y_pred,test_y)
print(score*100)
pickle.dump(best,open('D:\\project\\model.p','wb'))

