import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.impute import SimpleImputer
from statsmodels.tsa.stattools import adfuller
import warnings
from itertools import product
from prophet import Prophet
import numpy as np

def setup_google_sheets(creds_path, sheet_name, worksheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).worksheet(worksheet_name)

def prepare_data(data, headers):
    df = pd.DataFrame(data[1:], columns=headers)
    
    for col in headers[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('ME')  
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

def find_optimal_sarima_order(train_series, exog_train, max_order=2, verbose=False):
    warnings.filterwarnings('ignore')
    
    def check_stationarity(series):
        try:
            adf_result = adfuller(series, regression='ct')
            return adf_result[1] < 0.05
        except:
            return False

    d = 0
    D = 0
    temp_series = train_series.copy()
    
    while d <= 1 and not check_stationarity(temp_series):
        d += 1
        temp_series = temp_series.diff().dropna()
    
    temp_series = train_series.copy()
    while D <= 1 and not check_stationarity(temp_series.diff(12).dropna()):
        D += 1
        temp_series = temp_series.diff(12).dropna()
    
    p_range = range(max_order + 1)
    q_range = range(max_order + 1)
    P_range = range(max_order)
    Q_range = range(max_order)
    
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    total_combinations = len(list(product(p_range, q_range, P_range, Q_range)))
    if verbose:
        print(f"Testing {total_combinations} model combinations...")
    
    for p, q, P, Q in product(p_range, q_range, P_range, Q_range):
        try:
            model = SARIMAX(
                train_series,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            result = model.fit(disp=False, maxiter=500, method='nm')
            current_aic = result.aic
            
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, 12)
                
                if verbose:
                    print(f"New best model found:")
                    print(f"Order: {best_order}")
                    print(f"Seasonal Order: {best_seasonal_order}")
                    print(f"AIC: {best_aic}\n")
                
        except:
            continue
    
    return best_order, best_seasonal_order, best_aic

def improved_sarima_forecast(train_series, exog_train, exog_forecast, forecast_steps, verbose=False):
    try:
        best_order, best_seasonal_order, aic = find_optimal_sarima_order(
            train_series, 
            exog_train,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nFitting final model with:")
            print(f"Order: {best_order}")
            print(f"Seasonal Order: {best_seasonal_order}")
        
        model = SARIMAX(
            train_series, 
            exog=exog_train, 
            order=best_order,
            seasonal_order=best_seasonal_order
        )
        
        model_fit = model.fit(disp=False, maxiter=10000, method='nm')
        
        # Calculate and print variable importance
        if exog_train is not None and len(exog_train.columns) > 0:
            params = model_fit.params
            exog_params = {col: abs(params.get(col, 0)) for col in exog_train.columns}
            total_exog_importance = sum(exog_params.values())
            
            print("\nVariable Importance SARIMAX:")
            for col, param_value in sorted(exog_params.items(), key=lambda x: x[1], reverse=True):
                importance_percent = (param_value / total_exog_importance) * 100 if total_exog_importance > 0 else 0
                print(f"{col}: {importance_percent:.2f}%")
        
        forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog_forecast)
        predicted_mean = forecast.predicted_mean
        conf_int_80 = forecast.conf_int(alpha=0.2)
        conf_int_95 = forecast.conf_int(alpha=0.05)
        
        return predicted_mean, conf_int_80, conf_int_95
    
    except Exception as e:
        if verbose:
            print(f"Error in SARIMA forecasting: {str(e)}")
        return None, None, None
    
def prophet_forecast(train_series, exog_train, exog_forecast, forecast_steps):
    try:
        df_train = pd.DataFrame({
            'ds': train_series.index,
            'y': train_series.values
        })
        
        exog_train = exog_train.loc[train_series.index]
        for col in exog_train.columns:
            df_train[col] = exog_train[col].values
        
        model = Prophet()
        for col in exog_train.columns:
            model.add_regressor(col)
        model.fit(df_train)
        
        # Print regressor weights
        print("Regressor weights PROPHET (coefficients in original scale):")
        regressor_names = exog_train.columns.tolist()
        if regressor_names:
            # Get beta parameters; assuming first len(regressor_names) entries are the regressors
            beta = model.params['beta'][0]  # Assuming MAP estimation (only one sample)
            for i, col in enumerate(regressor_names):
                # Get standardization parameters
                std = model.extra_regressors[col]['std']
                # Compute coefficient in original scale (beta / std)
                coef = beta[i] / std if std != 0 else 0.0
                print(f"{col}: {coef:.4f}")
        else:
            print("No regressors used.")
        
        future = pd.DataFrame({'ds': exog_forecast.index})
        for col in exog_forecast.columns:
            future[col] = exog_forecast[col].values
        
        model.interval_width = 0.80
        forecast_80 = model.predict(future)
        conf_int_80 = forecast_80[['yhat_lower', 'yhat_upper']]
        
        model.interval_width = 0.95
        forecast_95 = model.predict(future)
        conf_int_95 = forecast_95[['yhat_lower', 'yhat_upper']]
        
        predicted_mean = forecast_95['yhat']
        predicted_mean.index = exog_forecast.index
        conf_int_80.index = exog_forecast.index
        conf_int_95.index = exog_forecast.index
        
        return predicted_mean, conf_int_80, conf_int_95
    except Exception as e:
        print(f"Error in Prophet forecasting: {str(e)}")
        return None, None, None
    
def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def backtest_models(observed_series, exog_data, initial_train_months=12, validation_months=6):
    """Perform time series cross-validation to determine model weights"""
    total_months = len(observed_series)
    weights = {'sarimax': 0, 'prophet': 0}
    errors = {'sarimax': [], 'prophet': []}
    
    for cutpoint in range(initial_train_months, total_months - validation_months, validation_months):
        train_end = observed_series.index[cutpoint]
        val_end = observed_series.index[cutpoint + validation_months]
        
        # Split data
        train = observed_series.loc[:train_end]
        val = observed_series.loc[train_end:val_end][1:]  # Avoid overlap
        exog_train = exog_data.loc[train.index]
        exog_val = exog_data.loc[val.index]
        
        # SARIMAX backtest
        try:
            sarima_pred, _, _ = improved_sarima_forecast(
                train, exog_train, exog_val, len(val)
            )
            if sarima_pred is not None:
                sarima_error = calculate_mape(val, sarima_pred)
                errors['sarimax'].append(sarima_error)
        except Exception as e:
            print(f"SARIMAX backtest failed at {train_end}: {str(e)}")
        
        # Prophet backtest
        try:
            prophet_pred, _, _ = prophet_forecast(
                train, exog_train, exog_val, len(val)
            )
            if prophet_pred is not None:
                prophet_error = calculate_mape(val, prophet_pred)
                errors['prophet'].append(prophet_error)
        except Exception as e:
            print(f"Prophet backtest failed at {train_end}: {str(e)}")
    
    # Calculate weights based on average errors
    avg_errors = {
        model: np.mean(errs) if errs else np.inf
        for model, errs in errors.items()
    }
    
    # Inverse weighting with smoothing
    total = sum(1/(avg_errors[model] + 1e-8) for model in avg_errors)
    weights = {
        model: (1/(avg_errors[model] + 1e-8)) / total
        for model in avg_errors
    }
    
    print(f"\nBacktesting Results:")
    print(f"SARIMAX Average MAPE: {avg_errors['sarimax']:.2f}%")
    print(f"Prophet Average MAPE: {avg_errors['prophet']:.2f}%")
    print(f"Final Weights - SARIMAX: {weights['sarimax']:.2f}, Prophet: {weights['prophet']:.2f}")
    
    return weights

def update_sheets_safe(sheet, data):
    cleaned_data = [[None if pd.isna(cell) else cell for cell in row] for row in data]
    
    try:
        sheet.batch_update([{
            'range': 'A1',
            'values': cleaned_data
        }])
    except Exception as e:
        print(f"Error updating Google Sheets: {str(e)}")

def main():
    CREDS_PATH = "C:\\Users\\nyter\\Desktop\\Scraper\\googleauth.json"
    headers = ['Date', 'InterestRate', 'ExchangeRate', 'UnemploymentRate', 
              'CPI', 'Inflation', 'ConsumerSpending', 'Sentiment', 'AmazonSales']
    feature_columns = headers[1:7]
    
    try:
        COUNTRY = input("Please choose the country by number. Options:\n(1) Egypt\n(2) Saudi\n(3) Turkey\n(4) UAE: \n")
        
        country_map = {"1": "EGYPT - IN DEVELOPMENT", "2": "SAUDI", "3": "TURKEY", "4": "UAE"}
        COUNTRY = country_map.get(COUNTRY, None)
        if not COUNTRY:
            print("Invalid country. Please try again.")
            exit()
        
        sheet = setup_google_sheets(CREDS_PATH, "Model", COUNTRY)
        data = sheet.get_all_values()
        
        df = prepare_data(data, headers)
        features_imputed = impute_features(df, feature_columns)
        target = df['AmazonSales']
        
        observed_sales = target.dropna()
        missing_sales_index = target[target.isna()].index
        
        if len(missing_sales_index) > 0:
            weights = backtest_models(observed_sales, features_imputed)
            sarima_mean, sarima_80, sarima_95 = improved_sarima_forecast(
                observed_sales,
                features_imputed.loc[observed_sales.index],
                features_imputed.loc[missing_sales_index],
                len(missing_sales_index)
            )
            
            prophet_mean, prophet_80, prophet_95 = prophet_forecast(
                observed_sales,
                features_imputed.loc[observed_sales.index],
                features_imputed.loc[missing_sales_index],
                len(missing_sales_index)
            )
            
            # Create weighted ensemble
            forecasts = []
            if sarima_mean is not None:
                forecasts.append(sarima_mean * weights['sarimax'])
            if prophet_mean is not None:
                forecasts.append(prophet_mean * weights['prophet'])
            
            if not forecasts:
                print("All models failed to forecast.")
                exit()
                
            final_forecast = pd.concat(forecasts, axis=1).sum(axis=1)
            final_forecast.name = 'AmazonSales'
            
            plt.figure(figsize=(12, 6))
            plt.plot(observed_sales.index, observed_sales.values,
                     label='Actual Sales', marker='o', linestyle='-', color='blue')
            
            plt.plot(missing_sales_index, final_forecast,
                     label='Combined Forecast', 
                     marker='s', linestyle='-.', color='orange')
            
            '''
            if sarima_80 is not None:
                plt.fill_between(missing_sales_index,
                               sarima_80.iloc[:, 0],
                               sarima_80.iloc[:, 1],
                               color='red', alpha=0.1, label='SARIMAX 80% CI')
            if prophet_80 is not None:
                plt.fill_between(missing_sales_index,
                               prophet_80['yhat_lower'],
                               prophet_80['yhat_upper'],
                               color='green', alpha=0.1, label='Prophet 80% CI')
            '''
                
            if sarima_95 is not None:
                plt.fill_between(missing_sales_index,
                               sarima_80.iloc[:, 0],
                               sarima_80.iloc[:, 1],
                               color='red', alpha=0.2, label='SARIMAX 95% CI')
            if prophet_95 is not None:
                plt.fill_between(missing_sales_index,
                               prophet_80['yhat_lower'],
                               prophet_80['yhat_upper'],
                               color='green', alpha=0.2, label='Prophet 95% CI')
            
            plt.xlabel('Month')
            plt.ylabel('Amazon Sales (BN Local Currency)')
            plt.title(f'Amazon Sales in {COUNTRY} | Model Weights: SARIMAX {weights["sarimax"]:.2f}, Prophet {weights["prophet"]:.2f}')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            output_df = df.reset_index()
            output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')
            output_data = [output_df.columns.values.tolist()] + output_df.values.tolist()
            
            while True:
                UPDATE = input("Update Google Sheet? (Y/N): ").upper()
                if UPDATE == "Y":
                    update_sheets_safe(sheet, output_data)
                    print("Sheet Updated")
                    break
                elif UPDATE == "N":
                    print("Sheet not updated.")
                    break
                else:
                    print("Invalid input. Enter Y or N.")
            
            print("\nPrediction Summary:")
            print("Predictions made:", len(missing_sales_index))
            print("\nLast 5 predictions:")
            print(df.loc[missing_sales_index, 'AmazonSales'].tail())
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()