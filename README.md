# House Price Prediction — Streamlit Sample App

An interactive Streamlit sample app for a house-price prediction project. It runs immediately in **demo mode**, so you can preview the interface without downloading a dataset or training a model. Demo estimates are illustrative, not trained ML predictions.

## Run the sample app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Enable machine-learning predictions

1. Add your `Housing.csv` file to `data/Housing.csv`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Train and evaluate the models:

   ```bash
   python train.py
   ```

4. Run `streamlit run app.py` again. The app will automatically load the saved model and display training metrics.

The dataset is expected to contain a `price` target and the property feature columns used in the app form. Check `train.py` and update the form if your dataset uses different columns.

## Deploy and get a shareable link

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app, choose this repository and branch, and set the app file to `app.py`.
4. Deploy. Streamlit will provide a public URL you can share.

## Important

Demo mode uses a simple illustrative formula so the app works without a trained model. It is not a real market estimate. A trained model's quality depends on the dataset and should not be used as formal property or financial advice.
