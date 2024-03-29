import numpy as np


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class ReplaceTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer replace elements of given columns by following given
    replacement rules. All other columns will remains unchanged.

        Parameters
        ----------
        replacement_dict : dict
            The dictionary of replacement rules. Keys are names of columns to
            change and corresponding values are the dictionary of replacement
            for this column.

        Attributes
        ----------
        X : Pandas DataFrame
            Original Pandas DataFrame.
              
        replacement_dict : dict
            The dictionary of replacement rules. Keys are names of columns to
            change and corresponding values are the dictionary of replacement
            for this column.
    """

    def __init__(self, replacement_dict):
        self.replacement_dict = replacement_dict
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Transform the Pandas DataFrame X using the replacement dictionary.

        Parameters
        ----------
        X : Pandas DataFrame
            The Pandas DataFrame to transform.

        Returns
        -------
        X_new : Pandas DataFrame
            The DataFrame resulting from the transformation. 
        """

        X_new = X.copy()
        
        for column, dictionary in self.replacement_dict.items():
            X_new[column].replace(to_replace = dictionary, inplace =True)

        return X_new


class OneHotEncodTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer create a DataFrame by one-hot encod some columns of a
    given DataFrame. The columns to encod are given as a list of sting when the
    transformer is instantiated. All other colums will remains unchanged.

        Parameters
        ----------
        feature_names : list of str
            The list of columns to encod.

        Attributes
        ----------
        X : Pandas DataFrame
            Original Pandas DataFrame.
            
        X_new : Pandas DataFrame
            Pandas DataFrame to return after encoding.
          
        categories : Dict of string
            A dictionary with column's names as keys. The corresponding value
            is the list of categories existing in the column for the DataFrame
            given when the fit method is called.
            
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
        Fill the categories dictionary. At every column name is associated the
        categories of X of existing in this columns of X.

        Parameters
        ----------
        X : Pandas DataFrame
            DataFrame to extract the differents categories which will be used
            for the encoding.
        """

        columns = X.columns

        for f in self.feature_names:
            if self._checkFeatureIn(f, columns):
                self.categories[f] = sorted(list(set(str(X[f].values))))

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
            feature_names by a one-hot encoding.
        """

        self.X = X
        self.X_new = self.X.copy()

        columns_of_X = X.columns

        for f in self.feature_names:

            if self._checkFeatureIn(f, self.categories) and \
               self._checkFeatureIn(f, columns_of_X):

                categories = self.categories[f]

                # Get the result of a one-hot encoding on the column X[f]
                # with categories obtained durring the fit.
                encoded_np = self._encodColumn(X[f], categories)

                # Create a DataFrame from the encoded numpy array.
                df_encoded = self._createEncodedDataFrame(encoded_np, f,
                                                          categories)

                # Remove the column X[f] from X.
                self._removeColumn(f)

                # Add the new encoded DataFrame to X_new.
                self._addEncoded(df_encoded)

        return self.X_new

    def _checkFeatureIn(self, f, container):
        """
        Check if the feature named f is in the container.
        
        Parameters
        ----------
        f : string
            Name of a feature.
        
        container : list of string or dict
            list of column's names or dictionary of column's names with associated
            possible categories.
        """
        
        if f not in container:
            message = "The feature " + f + " not in the given DataFrame."
            print(message)
            return False
        else:
            return True

    def _encodColumn(self, column, categories):
        """
        Use OneHotEncoder transformer to encod the given column.

        Parameters
        ----------
        column : a Pandas Series
            The column to encod.

        Returns
        -------
        encoded : a numpy array
            The numpy array given by the one-hot encoding of the given column.
            The number of row is the same that in the column, and the number of
            columns is the number of categories.
            
        categories : list of str
            The list of all values possible in the given column.
        """

        encoder = OneHotEncoder(handle_unknown='ignore')

        # Fit the encoder
        if 'nan' in categories:
            categories.remove('nan')
        values = np.array(categories).reshape((-1, 1))
        encoder.fit(values)

        # Transform the column
        numpy_column = column.to_numpy(copy=True)
        encoded = encoder.transform(numpy_column.reshape(-1, 1)).toarray()

        return encoded

    def _addEncoded(self, df_columns):
        """
        Add the DataFrame df_columns to X_new attribute.

        Parameters
        ----------
        df_columns : Pandas DataFrame
            DataFrame to add to the X_new attribute.
        """

        self.X_new = pd.concat([self.X_new, df_columns], axis=1)

    def _removeColumn(self, column_name):
        """
        Remove the column named column_name from the DataFrame X_new.

        Parameters
        ----------
        column_name : a str
            The name of the column to remove.
        """
        self.X_new = self.X_new.drop(column_name, axis=1)

    def _createEncodedDataFrame(self, array, column_name, categories):
        """
        Create a DataFrame corresponding to the one-hot encoding output.

        Parameters
        ----------
        array : a numpy array
            The encoded columns.
        column_name : a str
            The original name of the encoded column.
        categories : list of str
            The list of ordered different values of the encoded column.
        """

        names = [column_name + '_' + n for n in categories]
        df = pd.DataFrame(data=array, index=self.X.index,
                          columns=names)
        return df

    
# Very quick test
if __name__ == "__main__":
    

    data1 = {'test1': ['blick', 'lopl', 'blublu', 'blibli', 'blublu'],
             'test2': ['1', '9', '4', '6', '6'], }
    data2 = {'test1': ['blick', 'blublu', 'blublu', 'olip', 'lopl'],
             'test2': ['1', '9', '4', '7', '3'], }

    X_fit = pd.DataFrame(data=data1)
    X_transform = pd.DataFrame(data=data2)

    encoder = OneHotEncodTransformer(["test1"])
    encoder.fit(X_fit)
    X_encoded = encoder.transform(X_transform)
    print(X_encoded)

    # The test1 feature given in implementation not in the Dataframe to fit.
    encoder.fit(X_fit[["test2"]])

    # The test1 given in implementation is not in the DataFrame to transform.
    encoder.transform(X_transform[['test2']])