import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from helper import logger
from sklearn.decomposition import PCA

class DataPreprocessing:
    def __init__(self, data:pd.DataFrame):
        self.data = data
        
    def checkNans(self):
        logger.info(f"{self} is checking nans in dataframe")
        
        if(sum(self.data.isna().sum()) == 0):
            print("No nans in dataframe")
        else:
            print("Nans are found in dataframe")
            print("Please choose variant replace nans with mean of columns or drop them (drop or mean,median,most_frequent,constant) : ")
    
            strategies = ["mean",'median',"most_frequent",'constant']
            variant = input().lower()
            if variant in strategies:
                imputer = SimpleImputer(strategy=variant)
                return pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns, index=self.data.index)
            else:
                print("Invalid input. Returning original data.")
                return self.data
            
        logger.info(f"{self} is checking nans in dataframe ended")
        
    def scaling(self,target_column):
        logger.info(f"{self} is starting scaling")  
        
        print("Scaling with StandartScaler")      
        scaler = StandardScaler()
        scaler.fit(self.data.drop(target_column,axis=1))
        scaled_features = scaler.transform(self.data.drop(target_column, axis=1))
        updated = pd.DataFrame(scaled_features,columns=self.data.columns[:-1])
        updated[target_column] = self.data[target_column]
        print(updated.head())
        
        logger.info(f"{self} scaling ended") 
        
    def removeOutliers(self):
        logger.info(f"{self} is removing outliers from dataframe")

        cleaned_data = self.data.copy()
        for column in self.data.columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]

        logger.info(f"{self} removed outliers from dataframe")
        
        return cleaned_data
    
    def applyLogTransformation(self, column):
        logger.info(f"{self} is applying log transformation to {column}")

        transformed_data = self.data.copy()
        if (self.data[column] <= 0).any():
            logger.warning(f"Log transformation is not applied to {column} as it contains non-positive values.")
            return transformed_data
        else:
            transformed_data[column] = np.log(transformed_data[column])
            logger.info(f"Log transformation applied to {column}")
            return transformed_data
    
    def decomposition(self,scaled_data,n_components):
        logger.info(f"{self} is applying PCA decomposition to data")
        pca = PCA(n_components)
        print(pca.explained_variance_,"Explained variance")
        transformed_data = pca.fit_transform(scaled_data)
        explained_variance_ratio_sum = sum(pca.explained_variance_ratio_)
        print(f"Explained variance ratio: {explained_variance_ratio_sum * 100:.2f}%")
        logger.info(f"PCA decomposition ended")
        return transformed_data
            ### after this do scatter plot to understand the data 
        # plt.figure(figsize=(8,6))
        # plt.scatter(data[:,0],data[:,1], c=data["target_column"],cmap="plasma")
        # plt.xlabel("First pricipal component")
        # plt.ylabel("Second pricipal component")
        
        
    


                
            
        