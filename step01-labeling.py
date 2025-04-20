from datetime import datetime
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

ticker = "KO"
today = datetime.now().strftime('%Y-%m-%d')
data = yf.Ticker(ticker).history(start="2007-01-01", end=today)
df = data[["Open", "High", "Low", "Close"]].copy()

# Compute Range = High - Low
df['Range'] = df['High'] - df['Low']

# Compute Body Length = abs(Close - Open)
df['Body'] = abs(df['Close'] - df['Open'])

# Compute proportion of body length to range
df['BodyRangeRatio'] = df['Body'] / df['Range']

# Compute the direction of candle
threshold = 0.001
df['Direction'] = df.apply(
    lambda row: 2 if abs(row['Open'] - row['Close']) < threshold else (1 if row['Close'] > row['Open'] else 0),
    axis=1
)  # 2 = Doji, 1 = Bullish, 0 = Bearish

# Computer the center of mass (location of the body center in relation to the range)
df['CoM'] = (df[['Open', 'Close']].mean(axis=1) - df['Low']) / df['Range'] 

# Compute Proportion of candle upper wick to range
df['UpperWick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']

# Compute Proportion of candle lower wick to range
df['LowerWick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']

# Select just the first N rows to manually label
N = 250
training_df = df.head(N).copy()

# Loop through each row in the training dataframe
for idx, row in training_df.iterrows():
    single_candle = pd.DataFrame({
        'Open': [row['Open']],
        'High': [row['High']],
        'Low': [row['Low']],
        'Close': [row['Close']]
    }, index=[pd.to_datetime(row.name)])  # Use index for mplfinance

    # Generate the stat text
    direction_label = (
        'Bullish' if row['Direction'] == 1 else
        'Bearish' if row['Direction'] == 0 else
        'Doji'
    )
    stats_text = (
        f"Range: {row['Range']:.2f}\n"
        f"Body: {row['Body']:.2f}\n"
        f"Body/Range: {row['BodyRangeRatio']:.2f}\n"
        f"Center of Mass: {row['CoM']:.2f}\n"
        f"Direction: {direction_label}\n"
        f"Upper Wick: {row['UpperWick']:.2f}\n"
        f"Lower Wick: {row['LowerWick']:.2f}"
    )

    # Create the plot
    fig, axlist = mpf.plot(
        single_candle,
        type='candle',
        style='charles',
        returnfig=True,
        figratio=(6, 4),
        title=f"Index: {idx}",
        tight_layout=True
    )

    # Annotate stats on the figure
    axlist[0].text(0.05, 0.95, stats_text, transform=axlist[0].transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # Show and wait for input to move on
    plt.show()

    # Label the candlestick based on your own categorization paradigm
    training_df.loc[idx, 'Label'] = input("Enter label for this candle: ")

    # Save the labeled training data as a CSV
    training_df.to_csv(f"{ticker}-training.csv")

    # Optional: Allow break to stop early
    cont = input("Press [Enter] to continue, or type 'q' to quit: ")
    if cont.lower() == 'q':
        break