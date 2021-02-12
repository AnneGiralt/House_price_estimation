from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from warnings import warn
import pandas as pd
import numpy as np

import hashlib

def selectIntWithRatio(i, ratio, hash):
    """
    This function select or not an integer with a probability close to ratio by
    using the last byte of a hash of the integer i.

            Parameters:
                    i (int): An integer to select or not
                    ratio (float): A real number of [0,1] 
                    hash (hash constructor) : A hash constructor method

            Returns:
                    selected (bool) : Whether or not the integer i is selected
    """
    
    selected = hash(np.int64(i)).digest()[-1] < 256 * ratio
    
    return selected



def selectTestTrainSetId(id_list, test_ratio, hash = hashlib.md5):
    """
    This function split a list of integer into two list of integers, with a 
    given ratio using the last byte of a hash of each integer.

            Parameters:
                    id_list (list):A list of integer
                    test_ratio (float): A real number of [0,1], the putative 
                    ratio of the test set.
                    hash (hash constructor) : A hash constructor method

            Returns:
                    train_id (list) : integers to form the train set
                    test_id (list) : integers to form the test set
    """
    
    train_id = []
    test_id = []
    
    for i in id_list:
        if selectIntWithRatio(i, test_ratio, hash):
            test_id.append(i)
        else:
            train_id.append(i)
        
    return train_id, test_id 


class OneHotEncodTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer take Pandas DataFrame, a list of columns to encod and 
    return a pandas DataFrame with encoded columns.
    
        Parameters
        ----------
        feature_names : list of str
            The list of name of columns to encod.
        
        Attributes
        ----------
        X : Pandas DataFrame
            Original Pandas DataFrame.
        X_new : Pandas DataFrame
            Pandas DataFrame to return at the end.
        categories : dict of list of str
            A dictionary with column names as keys. A coresponding value will 
            be a list of catagories present in the coresponding column in the 
            DataFrame to fit.
        feature_names : list of str
            The list of name of columns to encod.
    """
    
    def __init__(self, feature_names):
        self.X = None
        self.X_new = None
        self.categories = {}
        self.feature_names = feature_names
    
    def fit(self, X, y=None): 
        """
        Fit the categories dictionary attribute by geting categories from 
        every columns of X of names in feature_name.
        
        Parameters
        ----------
        X : Pandas DataFrame
            DataFrame to extract the differents categories which will be used 
            for the encoding.
        """
        
        columns = X.columns
        
        for f in self.feature_names :
            if self._checkFeatureIn(f, columns):
                self.categories[f] = sorted(list(set(X[f].values)))

        return self
        
    
    def transform(self, X, y=None): 
        """
        Transform the Pandas DataFrame X by one-hot encod the given columns.
        
        Parameters
        ----------
        X : Pandas DataFrame
            The Pandas DataFrame to transform by encoding given columns.
        
        Returns
        -------
        self.X_new : Pandas DataFrame
            The DataFrame resulting from the transformation of every columns of 
            col_name by a one-hot encoding.        
        """
        
        self.X = X
        self.X_new = self.X.copy()

        
        columns_of_X = X.columns
        
        for f in self.feature_names :
            
            if self._checkFeatureIn(f, self.categories) and \
               self._checkFeatureIn(f, columns_of_X):
        
                categories = self.categories[f]
                
                # Get the result on a one-hot encoding on the colums X[f] 
                # with categories obtained by the fit method.
                encoded_np = self._encodColumn(X[f], categories)
                
                # Create a DataFrame from the encoded numpy array.
                df_encoded = self._createEncodedDataFrame(encoded_np, f, 
                                                          categories)
                
                # Remove the column [X[f]] from X.
                self._removeColumn(f)
            
                # Add the new encoded DataFrame to X_new.
                self._addEncoded(df_encoded)
                
        return self.X_new
    
    
    def _checkFeatureIn(self, f, container):
        if f not in container:
            message = "The feature " + f + " not in the given DataFrame." 
            print(message)
            return False
        else:
            return True
        
        
    def _encodColumn(self, column, categories):
        """
        Use the OneHotEncoder encoder to encod the given column.
        
        Parameters
        ----------
        columns : a pandas series 
            The column to encod.
            
        Returns
        -------
        encoded : a numpy array
            The numpy array given by the one-hot encoding of the given column.
            The number of row is the same that columns, and the number of 
            columns is the number of categories.
        categories : list of str
            The list of all values possible in the given column.
        """
        
        encoder = OneHotEncoder(handle_unknown ='ignore')
        
        # Fit the encoder
        values = np.array(categories).reshape((-1,1))
        encoder.fit(values)
        
        # Transform the column
        numpy_column = column.to_numpy(copy=True)
        encoded = encoder.transform(numpy_column.reshape(-1,1)).toarray()
        
        return encoded
    
    def _addEncoded(self, df_columns):
        """
        Add the DataFrame df_columns to X_new attribute.
        
        Parameters
        ----------
        df_columns : Pandas DataFrame
            DataFrame to add to the X_new attribute.
        """
        
        self.X_new = pd.concat([self.X_new,df_columns],axis=1)
        
    
    def _removeColumn(self, column_name):
        """
        Remove the column named column_name from the DataFrame X_new.
        
        Parameters
        ----------
        column_name : a str
            The name of the column to remove.
        """
        self.X_new = self.X_new.drop(column_name, axis=1)
    
    def _createEncodedDataFrame(self, array, col_name, categories):
        """
        Create a DataFrame corresponding to the one-hot encoding output.
        
        Parameters
        ----------
        array : a numpy array
            The encoded columns.
        col_name : a str
            The original name of the encoded column.
        categories : list of str
            The list of ordered different values of the encoded column.
        """
        
        names = [col_name + '_' + n for n in categories]
        df = pd.DataFrame(data = array, index = self.X.index, 
                                  columns = names)
        return df


   

if __name__=="__main__":
    
    data1 = {'test1' : ['blick', 'lopl', 'blublu', 'blibli', 'blublu'],
             'test2' : ['1', '9', '4', '6', '6'],}
    data2 = {'test1' : ['blick', 'blublu', 'blublu', 'olip', 'lopl'],
             'test2' : ['1', '9', '4', '7', '3'],}
    
    X_fit = pd.DataFrame(data = data1)
    X_transform = pd.DataFrame(data = data2)
    
    encoder = OneHotEncodTransformer(["test1"])
    encoder.fit(X_fit)
    X_encoded = encoder.transform(X_transform) 
    print(X_encoded)
    
    # The test1 feature given in implementation not in the Dataframe to fit.
    encoder.fit(X_fit[["test2"]])
                
    # The test1 given in implementation is not in the DataFrame to transform.
    encoder.transform(X_transform[['test2']])





