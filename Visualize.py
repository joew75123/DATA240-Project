import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
import seaborn as sns

num_name = [ 'host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count', 'accommodates', 'bedrooms', 'beds', 
'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_accuracy',
'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
'reviews_per_month', 'amenities']
cat_name = ['host_is_superhost', 'neighbourhood_cleansed', 'property_type', 'room_type', 'instant_bookable', 
'bathrooms_text', 'host_identity_verified', 'host_response_time']

class Visualize:
    def __init__(self):
        pass
    
    def check_num_corr(self,df):

        # Check the correlation between numeric variables
        plt.figure(figsize = (20, 20))
        sns.heatmap(df[num_name].corr(), annot = True, cmap="YlGnBu")
        plt.title("Correlation Matrix", fontsize = 30)
        plt.savefig("Figure/correlation.png")
        plt.close()
        # compute the correlation matrix
        corr_matrix = df[num_name].corr()

        #    find the indices of the highly correlated variables
        high_corr_indices = np.where(abs(corr_matrix) > 0.8)

        # print the highly correlated variables
        for i in range(len(high_corr_indices[0])):
            if high_corr_indices[0][i] < high_corr_indices[1][i]:
                print(f"{df[num_name].columns[high_corr_indices[0][i]]} and {df[num_name].columns[high_corr_indices[1][i]]} are highly correlated with a coefficient of {corr_matrix.iloc[high_corr_indices[0][i], high_corr_indices[1][i]]}.")
    """Drop certain columns using 0.8 as threshold
Drop reviews_per_month because it is highly correlated with numer_of_reviews, number_of_reviews_ltm, number_of_reviews_l30d

Drop calculated_host_listing_count because it is highly correlated with calculated_host_listing_count_private_homes

Drop beds and bedroom because they are highly correlated with accommodates

Drop availability_60 and availability_90.

review_scores_rating, review_scores_accuracy"""


    def distri_num_col(self,df):
        # Visualize using scatter plot and box plot
        df_num_name = df[num_name]
        for col in num_name:
            # Box plot
            sns.histplot(x=col, data=df_num_name)
            plt.title(f'{col.capitalize()} vs Price')
            plt.savefig(f"Figure/distribution_num_col/col_{col}.png")
            plt.close()
            
            
            
    def visual_num_col(self,df):
        # Visualize using scatter plot and box plot
        df_num_name = df[num_name + ['price']]
        for col in num_name:
            # Box plot
            sns.scatterplot(x=col, y='price', data=df_num_name)
            plt.title(f'{col.capitalize()} vs Price')
            plt.savefig(f"Figure/corr_num_col/col_{col}.png")
            plt.close()
            

    def visual_cat_col(self,df):
        # Create a new dataframe with only the selected columns
        df_cat_name = df[cat_name + ['price']]

        # Visualize using scatter plot and box plot
        for col in cat_name:
            # Box plot
            sns.boxplot(x=col, y='price', data=df_cat_name)
            plt.xticks(rotation=90)
            plt.title(f'{col.capitalize()} vs Price')
            plt.savefig(f"Figure/corr_cat_col/col_{col}.png")
            plt.close()