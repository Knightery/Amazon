import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.impute import SimpleImputer

def setup_google_sheets(creds_path, sheet_name, worksheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).worksheet(worksheet_name)

def prepare_data(data, headers):
    # Convert to DataFrame
    df = pd.DataFrame(data[1:], columns=headers)
    
    # Convert columns to numeric, coercing errors to NaN
    for col in headers[1:]:  # Skip the Date column
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set Date as index with frequency
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('ME')  # Set monthly frequency
    
    return df

def impute_features(df, feature_columns):
    features = df[feature_columns]
    imputer = SimpleImputer(strategy='mean')
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    return features_imputed

def sarima_forecast(train_series, exog_train, exog_forecast, forecast_steps):
    """
    Fits a SARIMA model with improved parameters and error handling
    """
    try:
        model = SARIMAX(train_series, 
                    exog=exog_train, 
                    order=(0, 2, 0),           # Simplified order
                    seasonal_order=(1, 0, 0, 12))  # Simplified seasonal order
        
        model_fit = model.fit(disp=False, 
                            maxiter=10000,
                            method='nm')  # Added optimization parameters
        
        forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog_forecast)
        return forecast.predicted_mean, forecast.conf_int()
    
    except Exception as e:
        print(f"Error in SARIMA forecasting: {str(e)}")
        return None, None

def update_sheets_safe(sheet, data):
    """
    Safely update Google Sheets by handling NaN values
    """
    # Replace NaN values with None (null in Google Sheets)
    cleaned_data = [[None if pd.isna(cell) else cell for cell in row] for row in data]
    
    try:
        # Use batch update for better performance
        sheet.batch_update([{
            'range': 'A1',
            'values': cleaned_data
        }])
    except Exception as e:
        print(f"Error updating Google Sheets: {str(e)}")

def main():
    # Configuration
    CREDS_PATH = "C:\\Users\\nyter\\Desktop\\Scraper\\googleauth.json"
    headers = ['Date', 'InterestRate', 'ExchangeRate', 'UnemploymentRate', 
              'CPI', 'Inflation', 'ConsumerSpending', 'Sentiment', 'AmazonSales']
    feature_columns = headers[1:7]  # Columns 2-7
    
    try:
        COUNTRY = input("Please choose the country by number. Options:\n1. Egypt\n2. Saudi\n3. Turkey\n4. UAE: \n")
        
        if COUNTRY == "1":
            COUNTRY = "EGYPT"
        elif COUNTRY == "2":
            COUNTRY = "SAUDI"
        elif COUNTRY == "3":
            COUNTRY = "TURKEY"
        elif COUNTRY == "4":
            COUNTRY = "UAE"
        else:
            print("Invalid country. Please try again.")
            exit()
        sheet = setup_google_sheets(CREDS_PATH, "Model", COUNTRY)
        data = sheet.get_all_values()
        
        # Prepare data
        df = prepare_data(data, headers)
        features_imputed = impute_features(df, feature_columns)
        target = df['AmazonSales']
        
        # Identify observed and missing sales
        observed_sales = target.dropna()
        missing_sales_index = target[target.isna()].index
        
        if len(missing_sales_index) > 0:
            # Generate forecasts
            forecast_sales, conf_intervals = sarima_forecast(
                train_series=observed_sales,
                exog_train=features_imputed.loc[observed_sales.index],
                exog_forecast=features_imputed.loc[missing_sales_index],
                forecast_steps=len(missing_sales_index)
            )
            
            if forecast_sales is not None:
                # Update DataFrame with predictions
                df.loc[missing_sales_index, 'AmazonSales'] = forecast_sales
                
                # Create visualization
                plt.figure(figsize=(12, 6))
                plt.plot(observed_sales.index, observed_sales.values, 
                        label='Actual Sales', marker='o', linestyle='-', color='blue')
                plt.plot(missing_sales_index, forecast_sales, 
                        label='Predicted Sales', marker='x', linestyle='--', color='red')
                
                if conf_intervals is not None:
                    plt.fill_between(missing_sales_index,
                                   conf_intervals.iloc[:, 0],
                                   conf_intervals.iloc[:, 1],
                                   color='red', alpha=0.1,
                                   label='95% Confidence Interval')
                
                plt.xlabel('Date')
                plt.ylabel('Amazon Sales')
                plt.title(f'Amazon Sales in {COUNTRY}: Actual vs Predicted')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                # Prepare and update Google Sheets
                output_df = df.reset_index()
                output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')
                output_data = [output_df.columns.values.tolist()] + output_df.values.tolist()
                while True:
                    UPDATE = input("Would you like to update the Google Sheet with this data? (Y/N): ")
                    if UPDATE.upper() == "Y":
                        update_sheets_safe(sheet, output_data)
                        print("Sheet Updated")
                        break
                    elif UPDATE.upper() == "N":
                        print("Sheet not updated.")
                        break
                    else:
                        print("Invalid answer. Please enter only Y or N.")
                
                # Print summary
                print("\nPrediction Summary:")
                print("Number of predictions made:", len(missing_sales_index))
                print("\nLast 5 predictions:")
                print(df.loc[missing_sales_index, 'AmazonSales'].tail())
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()