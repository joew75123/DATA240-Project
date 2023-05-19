import DataProcessor
import LoadDataset
import Visualize
import FeatureExtraction
import Modeling

def main():
    # Load Airbnb dataset
    airbnb_dataset = LoadDataset.LoadDataset('listings.csv')
    df = airbnb_dataset.load_data()

    # Create data processor and preprocess data
    data_processor = DataProcessor.DataProcessor()
    df=data_processor.process_price(df)
    df=data_processor.process_amenities(df)
    df=data_processor.process_num_col(df)
    
    Visualizer=Visualize.Visualize()
    Visualizer.check_num_corr(df)
    #Visualizer.distri_num_col(df)
    Visualizer.visual_num_col(df)
    Visualizer.visual_cat_col(df)
    
    df=data_processor.update_df(df)
    df=data_processor.drop_null(df)
    ##
    df=data_processor.stanardize_num_col(df)
    df=data_processor.encode_cat_col(df)
    
    extractor=FeatureExtraction.FeatureExtraction()
    df=extractor.feature_description_col(df)

    df=data_processor.normalize_price(df)
    X_train, X_test, y_train, y_test, X_val, y_val=data_processor.split_data(df)
    X_train,X_test,X_val=data_processor.PCA(X_train,X_test,X_val)
    
    Model=Modeling.Modeling()
    Model.Evaluate(X_train, X_test, y_train, y_test, X_val, y_val)


main()