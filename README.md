# Deck-Based Win Prediction in Clash Royale

### Data
The (large) dataset that I used for my project can be found here: https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset.  I used the data from the first day in the dataset--01/01/2021--which has about 3 million patterns.  As a toy model, I provided the top 100 patterns to run the codes with if the data fares too large.  It is important that you also download the "CardMasterList" file for preprocessing.

### Preprocessing
To preprocess the data, run:
>./preprocessor.py --data <raw_data_filepath> --cards <carddictionary_filepath> --output <newdata_filepath>
