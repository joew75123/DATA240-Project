import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


feature_name = ['host_acceptance_rate', 'host_response_rate', 'host_total_listings_count', 'accommodates', 
'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 
'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'reviews_per_month', 'amenities', 
'host_is_superhost', 'neighbourhood_cleansed', 'property_type', 'room_type', 'instant_bookable', 
'bathrooms_text', 'host_identity_verified', 'description', 'price', 'host_response_time', 'availability_30', 'availability_365']
numeric_cols = ['host_acceptance_rate', 'host_response_rate', 'host_total_listings_count', 'accommodates', 
'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating',
'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'reviews_per_month', 'amenities', 'availability_30', 'availability_365']
cat_name = ['host_is_superhost', 'neighbourhood_cleansed', 'property_type', 'room_type', 'instant_bookable', 
'bathrooms_text', 'host_identity_verified', 'host_response_time']

class DataProcessor:

    def __init__(self):
        pass

    def process_price(self, df):
        # process y - price
        df['price'] = df['price'].astype(str)
        df['price'] = df['price'].str.replace('$', '')
        df['price'] = df['price'].str.replace(',', '')
        df['price'] = pd.to_numeric(df['price'])
        # Plot a plot of the 'price' column
        plot = df['price'].plot().get_figure()
        plot.savefig("Figure/price_plot.png")
        # Create a boxplot of the 'y' column
        plt.boxplot(df['price'])
        # Add a title and labels to the plot
        plt.title('Boxplot of column price')
        plt.ylabel('Value')
        plt.savefig("Figure/price_boxplot.png")
        plt.close()
        
        # drop outlier
        # Calculate z-scores for the 'price' column
        z_scores = np.abs((df['price'] - df['price'].mean()) / df['price'].std())

        # Drop rows with z-scores greater than 3 (outliers)
        df = df[z_scores < 3]
        return df
    
    def process_amenities(self,df):
        df['amenities'] = df['amenities'].apply(len)
        return df
    

    
    def process_num_col(self, df):
        num_name = [ 'host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count', 'accommodates', 'bedrooms', 'beds', 
'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_accuracy',
'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
'reviews_per_month', 'amenities']
        # get rid of %
        df['host_response_rate'] = df['host_response_rate'].astype(str)
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '')
        df['host_acceptance_rate'] = df['host_acceptance_rate'].astype(str)
        df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '')

        df[num_name] = df[num_name].astype(float)
        return df
    
    def update_df(self,df):
        
        df = df[feature_name]
        return df

    def drop_null(self,df):
        df=df.dropna()
        return df
        
        
        
    def stanardize_num_col(self,df):
        # Select the numeric columns to standardize
        
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Standardize the selected columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def encode_cat_col(self,df):
        
        df = pd.get_dummies(df, columns=cat_name)
        return df
    
    def normalize_price(self,df):
        # normalize y
        scaler_price = StandardScaler()
        df['price'] = scaler_price.fit_transform(df[['price']])
        return df
    
    # Train and Validation, Test Split
    def split_data(self,df):
        # Split Train and Test
        np.random.seed(0)
        y = df.pop('price')
        df.pop('description')
        X=df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    #PCA to reduce dimension and update X_train, X_val and X_test
    def PCA(self,X_train,X_test,X_val):
        # Create PCA object and fit the data
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train)

        # Plot scree plot to visualize explained variance ratio of each component
        plt.plot(range(1, pca.n_components_+1), pca.explained_variance_ratio_)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.title('Scree plot')
        plt.savefig("Figure/PCA/PCA_plot.png")
        plt.close()
        
        # zoom in the scree plot
        plt.plot(range(1, 61), pca.explained_variance_ratio_[:60])
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.title('Scree plot (Zoom in)')
        plt.savefig("Figure/PCA/zoom_in_PCA_plot.png")
        plt.close()
        best_pc = 40
        pca = PCA(n_components=best_pc)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_val_pca = pca.transform(X_val)
        
        X_train = X_train_pca
        X_test = X_test_pca
        X_val = X_val_pca
        return X_train,X_test,X_val