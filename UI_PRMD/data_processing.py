import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

index_Spine_Base=0
index_Spine_Mid=4
index_Neck=8
index_Head=12   # no orientation
index_Shoulder_Left=16
index_Elbow_Left=20
index_Wrist_Left=24
index_Hand_Left=28
index_Shoulder_Right=32
index_Elbow_Right=36
index_Wrist_Right=40
index_Hand_Right=44
index_Hip_Left=48
index_Knee_Left=52
index_Ankle_Left=56
index_Foot_Left=60  # no orientation
index_Hip_Right=64
index_Knee_Right=68
index_Ankle_Right=72
index_Foot_Right=76   # no orientation
index_Spine_Shoulder=80
index_Tip_Left=84     # no orientation
index_Thumb_Left=88   # no orientation
index_Tip_Right=92    # no orientation
index_Thumb_Right=96  # no orientation

class Data_Loader():
    def __init__(self, dir):
        self.dir = dir
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 240
        self.new_label = []
        self.train_x,self.train_y= self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x,self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]
        return body_parts
    
    def import_dataset(self):
        x1 = pd.read_csv(os.path.join(self.dir, 'Data_Correct.csv'), header=None).values
        x2 = pd.read_csv(os.path.join(self.dir, 'Data_Incorrect.csv'), header=None).values
        y1 = pd.read_csv(os.path.join(self.dir, 'Labels_Correct.csv'), header=None).values
        y2 = pd.read_csv(os.path.join(self.dir, 'Labels_Incorrect.csv'), header=None).values

        x= np.concatenate((x1,x2),axis=0)
        y= np.concatenate((y1,y2),axis=0)

        x=x.reshape(-1, 39, 3, 240)
        x=x.transpose(0, 3, 1, 2)


        return x,y
            
    def preprocessing(self):
        X_reshaped = self.train_x.reshape(-1, self.train_x.shape[-1]*self.train_x.shape[-2])

        X_scaled = self.sc1.fit_transform(X_reshaped)
        X_scaled_reshaped = X_scaled.reshape(self.train_x.shape)


        y_train = self.sc2.fit_transform(self.train_y)


        return X_scaled_reshaped,y_train

