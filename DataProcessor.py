import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



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
        return df
    
    def process_amenities(self,df):
        df['amenities'] = df['amenities'].apply(len)
        return df
    

    
    def process_num_col(self, df):
        num_name = ['host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count', 'accommodates', 'bedrooms', 'beds', 
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
        feature_name = ['host_response_rate', 'host_acceptance_rate', 'host_total_listings_count', 'accommodates', 
                        'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 
                        'review_scores_accuracy',
                        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
                        'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'reviews_per_month', 'amenities', 
                        'host_is_superhost', 'neighbourhood_cleansed', 'property_type', 'room_type', 'instant_bookable', 
                        'bathrooms_text', 'host_identity_verified', 'description', 'price']
        df = df[feature_name]
        return df

    def drop_null(self,df):
        df=df.dropna()
        return df
        
        
        
    def stanardize_num_col(self,df):
        # Select the numeric columns to standardize
        numeric_cols = ['host_response_rate', 'host_acceptance_rate', 'host_total_listings_count', 'accommodates', 
        'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 
        'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
        'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'reviews_per_month', 'amenities']
        
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Standardize the selected columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def encode_cat_col(self,df):
        cat_name = ['host_is_superhost', 'neighbourhood_cleansed', 'property_type', 'room_type', 'instant_bookable', 
                    'bathrooms_text', 'host_identity_verified']
        df = pd.get_dummies(df, columns=cat_name)
        return df