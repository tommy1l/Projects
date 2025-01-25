import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Download Stock Data
def download_and_prepare_data(ticker, start_date, end_date, sample_size=100):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[["Open", "High", "Low", "Close", "Volume"]].dropna()  # Keep relevant columns
    # Add the Date column
    stock_data.reset_index(inplace=True)
    # Select the most recent data
    stock_data = stock_data.tail(sample_size)
    # Split into 80% train, 20% test
    train_data = stock_data.iloc[:int(len(stock_data) * 0.8)]
    test_data = stock_data.iloc[int(len(stock_data) * 0.8):]
    train_data.to_csv("stock-train.csv", index=False)
    test_data.to_csv("stock-test.csv", index=False)
    return "stock-train.csv", "stock-test.csv"

# Train the Model
def train_model(train_file):
    df = pd.read_csv(train_file)
    # Drop rows with missing values
    df = df.dropna()
    # Convert columns to numeric, make errors if NAN
    for col in ["High", "Low", "Open", "Volume", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with those errors
    df = df.dropna()
    # Scale the features
    scaler = StandardScaler()
    df[["High", "Low", "Open", "Volume"]] = scaler.fit_transform(df[["High", "Low", "Open", "Volume"]])
    features_key = ["High", "Low", "Open", "Volume"]
    iterations = 1000
    learning_rate = 0.01
    thetas = [0] * (len(features_key) + 1)

    # Gradient Descent
    for i in range(iterations):
        gradient = [0] * (len(features_key) + 1)
        for _, row in df.iterrows():
            feature_values = [1] + [row[feature] for feature in features_key]  # Include bias term
            y = row["Close"]
            thetaX = sum(thetas[j] * feature_values[j] for j in range(len(thetas)))
            for k in range(len(thetas)):
                gradient[k] += (y - thetaX) * (feature_values[k] if k > 0 else 1)
        for z in range(len(thetas)):
            thetas[z] += learning_rate * gradient[z] / len(df)
    print("Final Thetas:", thetas)
    return thetas, scaler

# Test the Model
def test_model(test_file, thetas, scaler):
    df = pd.read_csv(test_file)

    # Parse the Date column
    df["Date"] = pd.to_datetime(df["Date"])
    # Drop rows with NAN
    df = df.dropna()
    # Convert to int
    for col in ["High", "Low", "Open", "Volume", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop non-int rows
    df = df.dropna()
    # Use the scaler from training
    df[["High", "Low", "Open", "Volume"]] = scaler.transform(df[["High", "Low", "Open", "Volume"]])
    features_key = ["High", "Low", "Open", "Volume"]
    predicted = []
    for index, row in df.iterrows():
        # Include bias term
        feature_values = [1]
        # Add feature values from the row
        for feature in features_key:
            feature_values.append(row[feature])
        thetaX = 0
        for j in range(len(thetas)):
            thetaX += thetas[j] * feature_values[j]
        predicted.append(thetaX)
    return predicted, df["Close"].values, df["Date"]

# Step 4: Evaluate the Model
def evaluate_model(predicted, actual):
    # Ensure int arrays
    predicted = pd.to_numeric(predicted, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")
    mae = sum(abs(y - pred) for y, pred in zip(actual, predicted)) / len(actual)
    print(f"Mean Absolute Error: {mae}")
    return mae

# Plot Predicted vs Actual
import matplotlib.dates as matdates

def plot_predictions(predicted, actual, dates):
    plt.figure(figsize=(12, 6))

    # Plot actual and predicted prices
    plt.plot(dates, actual, label="Actual Prices", color="blue")
    plt.plot(dates, predicted, label="Predicted Prices", color="orange")
    plt.title("Predicted vs Actual Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    # Format x-axis in date format
    plt.gca().xaxis.set_major_formatter(matdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(matdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    # edit y-axis to include commas
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    plt.show()

# Main Program
if __name__ == "__main__":
    train_file, test_file = download_and_prepare_data(
        ticker="^GSPC", start_date="2022-01-01", end_date="2023-01-01", sample_size=100
    )

    thetas, scaler = train_model(train_file)
    predicted, actual, dates = test_model(test_file, thetas, scaler)
    evaluate_model(predicted, actual)
    plot_predictions(predicted, actual, dates)
