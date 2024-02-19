import pandas as pd
from helper import logger
from sklearn.decomposition import PCA

class PCA:
    def __init__(self, data:pd.DataFrame):
        self.data = data
        
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
        
        
    


                
            
        