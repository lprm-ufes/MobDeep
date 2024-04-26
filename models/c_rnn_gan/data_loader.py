"""
This code was based on the code music_data_utils, wryten by Olof Mogren, http://mogren.one/
in the project: https://github.com/olofmogren/c-rnn-gan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from numpy import load
from sklearn.preprocessing import MinMaxScaler

class DataLoader(object):
    def __init__(self, datadir, validation_percentage, test_percentage,filename="", n_samples=None, n_features=None, n_steps=None,scale=False):        
        self.pointer = {}
        self.datadir = datadir
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_steps = n_steps

        self.read_data(filename,validation_percentage,test_percentage,scale)

    def read_data(self,filename,val_percentage=None,test_percentage=None,scale=False):
        """
        Loads the dataset that is in the datadir directory provided

        Args:
            filename: the dataset's name (i.e. data.npy)
            val_percentage: percentage of the data that will be used as validation
            test_percentage: percentage of the data that will be used as test
            scale: indicates if the data should be scaled or not (if data is not in [0,1] range
            it should be scaled)

        Returns a array with dimensions [n_samples,n_steps,n_features]

        """
        print("filename, {}".format(filename))
        self.data = load(self.datadir+filename)
        if (self.data.dtype=='int64'):
          self.data = self.data.astype('float64')
        print("self.data ",self.data[0][0], scale)
        if (scale == True):
            print("Scaling")
            self.data = self.scale_data(self.data)
    
        valid_shape = (self.n_samples,self.n_steps,self.n_features)
        if self.data.shape!= valid_shape:
            print("Reshaping...")
            try:
                self.data = self.data.reshape(valid_shape)
            except Exception as e:                
                
                raise ValueError ("The data with shape {} couldn't be reshaped to {}, please provide a valid shape.".format(self.data.shape, valid_shape))                        
                exit()

        self.songs = {}
        self.songs['validation'] = []
        self.songs['test'] = []
        self.songs['train']  = []
        
        train_len = len(self.data)
        test_len = 0
        validation_len = 0
        
        # TODO criar excpetions para essa parte
        if val_percentage or test_percentage:
            if val_percentage:
                validation_len = int(float(val_percentage/100)*len(self.data))
                train_len = train_len - validation_len
            if test_percentage:
                test_len = int(float(test_percentage/100)*len(self.data))
                train_len = train_len - test_len
            self.songs['train'] = self.data[:train_len]
            self.songs['test'] = self.data[train_len:train_len+test_len]
            self.songs['validation'] = self.data[train_len+test_len:]

        else:
            self.songs['train'] = self.data
            self.songs['test']  = self.data
            self.songs['validation'] = self.data
        print("songs_train len: ",len(self.songs['train']))
        # pointers
        self.pointer['validation']  = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

    def get_batch(self,batch_size,part='train'):
        """
        Returns a batch
        Args:
            batch_size: the number of samples that will be drawed from the original data

        Returns a array with the dimension [batch_size,self.num_steps,self.num_features]
        """

        if (self.pointer[part]>len(self.songs[part])-batch_size):
            return [None,None]

        if len(self.songs[part])>0:

            start = self.pointer[part]
            end = self.pointer[part]+batch_size

            batch = self.songs[part][start:end]
            meta = np.random.randn(batch_size,1)
            self.pointer[part]+=batch_size        
            return [meta,batch]

        else:
            raise 'get_batch() called but self.songs is not initialized.'
    
    def get_num_features(self):
        """
        Returns the number of features
        """
        return self.n_features

    def get_num_meta_features(self):
        # just for test purposes
        return 1
    
    def rewind(self,part='train'):
        """
        Reset the pointer for the 'part'        
        """
        self.pointer[part] = 0
        
    def scale_data(self,data):
        """
        Scale the data
        Args:
            data: data to be scaled using the MinMaxScaler
        Returns:
            The data scaled
        """

        scaler = MinMaxScaler()
        data = data.reshape(-1,1)
        scaler = scaler.fit(data)
        data  = scaler.transform(data)

        return data