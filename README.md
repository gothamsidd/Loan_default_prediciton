# Loan Default Prediction

A machine learning project for predicting loan defaults using the Kaggle Home Credit dataset. Built a Streamlit dashboard and FastAPI backend to make predictions and explore the data.

## Getting Started

### 1. Download the Dataset

The dataset files are too large for GitHub. Download them from Kaggle:

- Go to: https://www.kaggle.com/c/home-credit-default-risk/data
- Download `application_train.csv` and `application_test.csv`
- Place them in the project root directory

See [SETUP_DATASET.md](SETUP_DATASET.md) for detailed instructions.

Install the dependencies:
```bash
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser. Use the sidebar to navigate between different pages.

## What's Included

**Streamlit Dashboard**
- Data exploration with interactive charts
- Model training (Logistic Regression, Decision Tree, KNN, XGBoost, LightGBM)
- Model comparison and evaluation
- Hyperparameter tuning
- Ensemble methods (Voting, Stacking, Blending)
- Single and batch predictions
- Model explainability (SHAP, feature importance)
- Threshold optimization
- Profit curve analysis

**FastAPI Backend**
- REST API for predictions
- Health check endpoint
- Model info endpoint
- Auto-generated docs at `/docs`

## Using the App

1. Start with the Overview page to see dataset statistics
2. Explore the data in Data Exploration
3. Train models in Model Training (enable "Use Full Dataset" first)
4. Compare models in Model Comparison
5. Make predictions in the Predictions page
6. Use Batch Predictions to process CSV files

## API Usage

Start the API:
```bash
uvicorn api:app --reload
```

Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amt_income_total": 200000.0,
    "amt_credit": 400000.0,
    "amt_annuity": 25000.0,
    "age": 35.0,
    "employ_years": 5.0
  }'
```

## Deployment

### Quick Deploy (Streamlit Cloud - Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy
4. Takes about 5 minutes

**Note:** For large datasets, see [DEPLOYMENT.md](DEPLOYMENT.md) for cloud storage options.

### Local Docker

Build and run with Docker:
```bash
docker-compose up -d
```

This starts both the Streamlit app and FastAPI service.

### Other Platforms

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- Railway
- Render
- Heroku
- AWS/GCP
- Self-hosting

## Project Structure

- `app.py` - Main Streamlit application
- `api.py` - FastAPI REST API
- `preprocessing.py` - Data preprocessing functions
- `config.yaml` - Configuration settings
- `utils/` - Utility modules (logging, config, validators, metrics, feature selection)
- `tests/` - Unit and integration tests
- `models/` - Saved model files and metadata

## Model Performance

Baseline results on test set:

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.74 | 0.92 |
| Decision Tree | 0.68 | 0.92 |
| XGBoost (tuned) | 0.78 | 0.92 |
| LightGBM (tuned) | 0.78 | 0.92 |

## Testing

Run tests:
```bash
pytest tests/ -v
```

## Notes

- The dataset is large (300K+ records), so the app uses sampling for faster exploration
- Enable "Use Full Dataset" in the sidebar when training models
- Models are saved to the `models/` directory with metadata
- Logs are written to `logs/` directory

## Requirements

Python 3.9+, see `requirements.txt` for full list of dependencies.
