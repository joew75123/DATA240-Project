import DataProcessor
import LoadDataset
import Visualize


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
    Visualizer.visual_num_col(df)
    Visualizer.visual_cat_col(df)
    
    df=data_processor.update_df(df)
    df=data_processor.drop_null(df)
    #print(df['amenities'])
    
    
    # airbnb_predictor = AirbnbPricePredictor(df, data_processor, MachineLearningModel())

    # airbnb_predictor.preprocess_data()

    # # Train machine learning model
    # airbnb_predictor.train_model()

    # # Evaluate model
    # score = airbnb_predictor.evaluate_model()
    # print(f'SVM R2 score: {score:.2f}')


main()