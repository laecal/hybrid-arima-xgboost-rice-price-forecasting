HYBRID ARIMA-XGBOOST FOR FARMGATE RICE CROP PRICE FORECASTING WEB SIMULATOR
Version 1.0
	
Prepared by:
Leahlyn Calacasan
Nina Rosalinda Arlos
Jhona Shane Hosillos
Shane Nadynn Noblezada

BSCS 4-B
College of Computing and Informatics
Iloilo Science and Technology University
Lapaz, Iloilo City

July 2026

	
1. SYSTEM OVERVIEW
This web simulator generates forecasts of farmgate rice prices in the Philippines
using a Hybrid ARIMA-XGBoost forecasting approach. It uses historical rice price
data and, when applicable, macroeconomic variables such as GDP growth and
inflation. Forecasts are presented through tables and charts.

2. REQUIREMENTS
- Python development environment (PyCharm is used for the backend)
- Required Python libraries installed
- Modern web browser (Google Chrome, Microsoft Edge, or Mozilla Firefox)
- Historical farmgate rice price CSV
- Optional macroeconomic CSV
- Internet connection, if required by the generated web link

3. REQUIRED DATA FILES
Historical Farmgate Rice Price Dataset:
    final-interpolated-farmgate-price.csv

Optional Macroeconomic Dataset:
    combined-gdp-inflation.csv

CSV files must follow the format expected by the simulator.

4. STARTING THE SIMULATOR
1. Open the project folder.
2. Open the terminal/command prompt.
3. Activate the required Python environment, if applicable.
4. Run the backend script.
5. Wait for the server to start successfully.
6. Copy the generated/static URL.
7. Open the URL in a supported web browser.

IMPORTANT: Keep the backend running while using the web simulator. If the
backend is stopped, the generated web link may no longer be accessible.

5. GENERATING A FORECAST
1. Upload the required historical Farmgate Price CSV.
2. Upload the Macro CSV if needed.
3. Select the number of months to forecast.
4. Choose whether to apply the same macroeconomic values to all forecast months.

IF YES:
- Enter the Inflation value.
- Enter the GDP growth value.
- Click Generate Forecast.

IF NO:
- Enter the expected Inflation and GDP values for each forecast month.
- Enter Actual Price values only when reliable actual prices are available.
- Click Generate Forecast.

6. FORECAST RESULTS
The results table may contain:
- Month - forecasted month.
- ARIMA Forecast - forecast from the ARIMA model.
- Hybrid Forecast - forecast from the Hybrid ARIMA-XGBoost model.
- Actual - actual rice price, when available.

The system may also display a comparison chart.

7. ERROR METRICS
When actual prices are provided, the system may calculate:
- RMSE - magnitude of prediction errors, with greater weight on larger errors.
- MAE - average absolute difference between forecasted and actual values.
- MAPE - average percentage difference between forecasted and actual values.
- MPE - average percentage error and tendency to overestimate or underestimate.

Lower error values generally indicate smaller forecasting errors. Interpret the
metrics together rather than relying on only one metric.

8. DM/HLN STATISTICAL TEST
When sufficient actual observations are available, the system compares ARIMA and
Hybrid ARIMA-XGBoost using the Diebold-Mariano test with the
Harvey-Leybourne-Newbold adjustment.

The test may indicate:
- Hybrid is better, but the difference is not statistically certain.
- Hybrid model is significantly more accurate than ARIMA.
- ARIMA is significantly better; Hybrid may be overfitting.
- Both models are performing similarly.

The test may not be computed when there are too few observations.

9. CLEARING INPUTS
Click Clear to remove the current inputs and reset the form.

10. CLOSING THE SYSTEM
1. Close the browser tab/window.
2. Return to the terminal or development environment running the backend.
3. Stop/terminate the backend process.

There is no logout procedure because the simulator does not require a user
account or login session.

11. TROUBLESHOOTING
Web simulator cannot be opened:
    Restart the backend and open the newly generated web link.

CSV cannot be uploaded:
    Make sure the file is a valid CSV and follows the required format.

Uploaded data is incorrect:
    Check for missing, invalid, or incorrectly formatted values.

Forecast cannot be generated:
    Check the forecast period and required macroeconomic inputs.

GDP/Inflation cannot be processed:
    Enter valid numerical values.

Error metrics are not displayed:
    Provide the corresponding Actual Price values and generate the forecast again.

DM/HLN test is not displayed:
    There may not be enough forecast observations and actual prices for the test.

Actual values are not shown:
    This is expected when actual prices for future months are not yet available.

Chart has no Actual line:
    Enter Actual Price values when reliable values are available.

System stops responding:
    Check whether the backend is still running. Restart it if necessary.

12. IMPORTANT REMINDERS
- Keep the backend script running while using the simulator.
- Use the correct generated web link.
- Upload CSV files in the required format.
- Enter macroeconomic values correctly.
- Enter actual prices only when reliable actual values are available.
- If problems persist, check the terminal/development environment for error messages.
