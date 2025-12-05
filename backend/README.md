# MigraineMamba Backend API

AI-Powered Migraine Prediction System using Mamba Architecture with Self-Supervised Learning.

## 🚀 Features

- **3-Phase Prediction System**
  - Foundation Model (Days 1-14): LightGBM-based instant predictions
  - Temporal Model (Days 15-30): Mamba architecture for sequence analysis
  - Personalized Model (Day 31+): Fine-tuned to individual triggers

- **Comprehensive Trigger Analysis**
  - Clinical odds ratios for 10+ trigger categories
  - Pattern discovery algorithms
  - Weekly accuracy tracking

- **RESTful API**
  - User onboarding and profile management
  - Daily log submission
  - Real-time predictions
  - Insights and analytics

## 📋 Prerequisites

- Python 3.10+
- pip or poetry

## 🛠️ Installation

1. **Clone and navigate to backend**
   ```bash
   cd migrainemamba-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📚 API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Main Endpoints

#### Users
- `POST /api/v1/users/onboarding` - Create new user
- `GET /api/v1/users/profile/{user_id}` - Get user profile
- `PUT /api/v1/users/profile/{user_id}` - Update profile

#### Daily Logs
- `POST /api/v1/logs/submit` - Submit daily log
- `GET /api/v1/logs/history/{user_id}` - Get log history
- `PUT /api/v1/logs/outcome/{user_id}/{date}` - Update migraine outcome

#### Predictions
- `GET /api/v1/predictions/{user_id}` - Get today's prediction
- `GET /api/v1/predictions/history/{user_id}` - Prediction history
- `GET /api/v1/predictions/accuracy/{user_id}` - Accuracy stats

#### Insights
- `GET /api/v1/insights/triggers/{user_id}` - Trigger analysis
- `GET /api/v1/insights/weekly-stats/{user_id}` - Weekly statistics
- `GET /api/v1/insights/recommendations/{user_id}` - Personalized recommendations

## 🧠 ML Models

### Foundation Model (LightGBM)
Uses clinical odds ratios for transparent predictions:
- Sleep Deficit: OR 3.98
- High Stress: OR 2.67
- Menstrual Phase: OR 2.04
- Poor Sleep Quality: OR 2.15
- Alcohol (5+ drinks): OR 2.08
- Skipped Meals: OR 1.89
- Bright Light: OR 1.54
- Dehydration: OR 1.45

### Temporal Model (Mamba)
Analyzes 14-day sequences for pattern recognition:
- Bidirectional state space modeling
- Self-supervised pre-training
- Temporal attention mechanisms

### Personalized Model
Fine-tuned with user-specific data:
- Individual trigger sensitivity weights
- Weekly retraining cycles
- Continuous learning from outcomes

## 📊 Database Schema

```
users
├── id (UUID)
├── gender, age, height, weight, bmi
├── location_city
├── attacks_per_month
├── has_menstrual_cycle, cycle_start_day
├── current_phase, days_logged
└── model_version

daily_logs
├── user_id (FK)
├── date
├── sleep_hours, sleep_quality_good
├── stress_level
├── skipped_meals, had_snack
├── alcohol_drinks, caffeine_drinks, water_glasses
├── bright_light_exposure, screen_time_hours
├── symptoms (JSON)
├── migraine_occurred, migraine_severity, etc.
└── predicted_probability, prediction_was_correct

predictions
├── user_id (FK)
├── prediction_date
├── attack_probability, risk_level, confidence
├── model_version, model_type
├── top_triggers, recommendations (JSON)
└── actual_outcome, was_correct
```

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | SQLite |
| `SECRET_KEY` | App secret key | - |
| `CORS_ORIGINS` | Allowed origins | localhost:3000 |
| `WEATHER_API_KEY` | OpenWeatherMap API key | - |
| `LOW_RISK_THRESHOLD` | Low risk cutoff | 0.3 |
| `MODERATE_RISK_THRESHOLD` | Moderate risk cutoff | 0.5 |
| `HIGH_RISK_THRESHOLD` | High risk cutoff | 0.7 |

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
migrainemamba-backend/
├── main.py                 # FastAPI application
├── requirements.txt
├── .env.example
├── app/
│   ├── api/               # API routes
│   │   ├── users.py
│   │   ├── predictions.py
│   │   ├── logs.py
│   │   └── insights.py
│   ├── core/              # Core configuration
│   │   ├── config.py
│   │   └── database.py
│   ├── models/            # Data models
│   │   ├── database.py    # SQLAlchemy models
│   │   └── schemas.py     # Pydantic schemas
│   ├── services/          # Business logic
│   │   ├── prediction_service.py
│   │   └── trigger_analysis_service.py
│   └── ml/                # ML models (placeholder)
└── tests/
```

## 🚀 Deployment

### Docker
```bash
docker build -t migrainemamba-api .
docker run -p 8000:8000 migrainemamba-api
```

### Production
1. Use PostgreSQL instead of SQLite
2. Set proper SECRET_KEY and JWT_SECRET_KEY
3. Configure CORS for your domain
4. Use gunicorn with uvicorn workers:
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request