# Random Forest Classifier for Candlesticks
This is an implementation of the Random Forest Classifier from scikit-learn used to classify daily candlesticks in stock market candlestick charts.

To use this package, manual labeling is required. I've made it somewhat manageable in the `step01-labeling.py` file. You will need your own paradigm for classifying candlesticks in order to label them appropriately.

## Manual Labeling
Use the following to manually label the candlestick data:
```
python step01-labeling.py
```

The file defaults to pulling in daily OHLC data for Coca Cola (KO) from 2007 to present day. Select the number of observations you wish to label using `N`. Then run the script. You will need to visually inspect each candle, exit the plot, and enter the label for that candle. Hit enter twice to move on to the next candle to label. At the end a CSV file is generated with the labeled data.

## Training the Model
Use the following to train a Random Forest Classifier on your labeled data:

```
python step02-training.py
```

The file fits a Random Forest Classifier with the labeled data in the generated CSV file. A classification report is generated with basic values for Precision, Recall, F1-Score and Support. Additionally, a plot with Feature Importances is generated for your review. At the end of the file, .pkl files are generated that represent the generated model.

To use the .pkl file, run the following:

```
model = joblib.load("candlestick_model.pkl")
```
where `candlestick_model.pkl` is the pkl file to load.

