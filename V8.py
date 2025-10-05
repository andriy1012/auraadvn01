import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Advanced - By Andriy",
    page_icon="📈",
    layout="wide"
)

st.title("🎯 Prediksi Harga Advanced - Lebih Akurat!")
st.markdown("Dengan formula technical analysis dan ensemble learning")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV data saham", type=['csv'])

def calculate_technical_indicators(df, price_col='Close'):
    """Hitung semua technical indicators"""
    df_tech = df.copy()

    # 1. MOVING AVERAGES
    df_tech['SMA_5'] = df_tech[price_col].rolling(5).mean()
    df_tech['SMA_10'] = df_tech[price_col].rolling(10).mean()
    df_tech['SMA_20'] = df_tech[price_col].rolling(20).mean()
    df_tech['SMA_50'] = df_tech[price_col].rolling(50).mean()

    # 2. EXPONENTIAL MOVING AVERAGE
    df_tech['EMA_12'] = df_tech[price_col].ewm(span=12).mean()
    df_tech['EMA_26'] = df_tech[price_col].ewm(span=26).mean()
    df_tech['EMA_50'] = df_tech[price_col].ewm(span=50).mean()

    # 3. MACD (Moving Average Convergence Divergence)
    df_tech['MACD'] = df_tech['EMA_12'] - df_tech['EMA_26']
    df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9).mean()
    df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['MACD_Signal']

    # 4. RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df_tech['RSI_14'] = calculate_rsi(df_tech[price_col], 14)
    df_tech['RSI_7'] = calculate_rsi(df_tech[price_col], 7)

    # 5. BOLLINGER BANDS
    df_tech['BB_Middle'] = df_tech[price_col].rolling(20).mean()
    bb_std = df_tech[price_col].rolling(20).std()
    df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * 2)
    df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * 2)
    df_tech['BB_Width'] = df_tech['BB_Upper'] - df_tech['BB_Lower']
    df_tech['BB_Position'] = (df_tech[price_col] - df_tech['BB_Lower']) / (df_tech['BB_Upper'] - df_tech['BB_Lower'])

    # 6. VOLATILITY MEASURES
    df_tech['volatility_5'] = df_tech[price_col].pct_change().rolling(5).std()
    df_tech['volatility_10'] = df_tech[price_col].pct_change().rolling(10).std()
    df_tech['volatility_20'] = df_tech[price_col].pct_change().rolling(20).std()

    # 7. TRUE RANGE & AVERAGE TRUE RANGE (ATR)
    high_low = df_tech['High'] - df_tech['Low'] if 'High' in df_tech.columns else df_tech[price_col] - df_tech[price_col].shift(1)
    high_close = np.abs((df_tech['High'] if 'High' in df_tech.columns else df_tech[price_col]) - df_tech[price_col].shift(1))
    low_close = np.abs((df_tech['Low'] if 'Low' in df_tech.columns else df_tech[price_col]) - df_tech[price_col].shift(1))

    df_tech['TR'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df_tech['ATR_14'] = df_tech['TR'].rolling(14).mean()

    # 8. PRICE MOMENTUM
    df_tech['momentum_5'] = df_tech[price_col] / df_tech[price_col].shift(5) - 1
    df_tech['momentum_10'] = df_tech[price_col] / df_tech[price_col].shift(10) - 1
    df_tech['rate_of_change'] = df_tech[price_col].pct_change(1)

    # 9. VOLUME INDICATORS (jika ada volume)
    if 'Volume' in df_tech.columns:
        df_tech['volume_sma_10'] = df_tech['Volume'].rolling(10).mean()
        df_tech['volume_ratio'] = df_tech['Volume'] / df_tech['volume_sma_10']
        df_tech['OBV'] = (df_tech['Volume'] * (~df_tech[price_col].diff().le(0) * 2 - 1)).cumsum()

    # 10. LAG FEATURES (untuk time series)
    for lag in [1, 2, 3, 5, 7, 10, 14]:
        df_tech[f'price_lag_{lag}'] = df_tech[price_col].shift(lag)

    # 11. ROLLING STATISTICS
    for window in [3, 5, 7, 10, 14]:
        df_tech[f'rolling_mean_{window}'] = df_tech[price_col].rolling(window).mean()
        df_tech[f'rolling_std_{window}'] = df_tech[price_col].rolling(window).std()
        df_tech[f'rolling_min_{window}'] = df_tech[price_col].rolling(window).min()
        df_tech[f'rolling_max_{window}'] = df_tech[price_col].rolling(window).max()

    # 12. TIME-BASED FEATURES
    if 'date_processed' in df_tech.columns and pd.api.types.is_datetime64_any_dtype(df_tech['date_processed']):
        df_tech['day_of_week'] = df_tech['date_processed'].dt.dayofweek
        df_tech['day_of_month'] = df_tech['date_processed'].dt.day
        df_tech['week_of_year'] = df_tech['date_processed'].dt.isocalendar().week
        df_tech['month'] = df_tech['date_processed'].dt.month
        df_tech['quarter'] = df_tech['date_processed'].dt.quarter

        # Seasonal dummies
        df_tech['is_monday'] = (df_tech['day_of_week'] == 0).astype(int)
        df_tech['is_friday'] = (df_tech['day_of_week'] == 4).astype(int)
        df_tech['month_end'] = (df_tech['day_of_month'] >= 25).astype(int)

    return df_tech

def create_advanced_models():
    """Buat advanced models dengan optimized parameters"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    return models

def create_voting_ensemble(trained_models):
    """Buat voting ensemble dari models yang sudah ditraining"""
    estimators = [(name, model) for name, model in trained_models.items()]
    voting_reg = VotingRegressor(estimators=estimators)
    return voting_reg

def time_series_cross_validate(model, X, y, n_splits=5):
    """Cross validation khusus time series"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)

    return np.mean(scores), np.std(scores)

def calculate_feature_importance(X, y):
    """Hitung feature importance menggunakan Random Forest"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    return importance_df

if uploaded_file is not None:
    try:
        # Baca file CSV
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ File berhasil diupload! {len(df)} baris data")

        # Tampilkan data
        st.subheader("📋 Data yang Diupload")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", len(df))
        with col2:
            st.metric("Total Kolom", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Kolom Numerik", len(numeric_cols))
        with col4:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'tanggal' in col.lower()]
            st.metric("Kolom Tanggal", len(date_cols))

        st.dataframe(df, use_container_width=True, height=500)

        # Konfigurasi data
        st.subheader("🔧 Konfigurasi Data")

        col1, col2 = st.columns(2)

        with col1:
            date_col = st.selectbox("Pilih kolom tanggal:", df.columns)

        with col2:
            price_col = st.selectbox("Pilih kolom harga:", numeric_cols)

        # Konversi tanggal
        try:
            df['date_processed'] = pd.to_datetime(df[date_col])
            df = df.sort_values('date_processed').reset_index(drop=True)
            use_date = True
        except:
            st.warning("⚠️ Tidak bisa konversi tanggal, menggunakan index")
            df['date_processed'] = range(len(df))
            use_date = False

        # Advanced Configuration
        st.subheader("⚙️ Konfigurasi Advanced")

        col1, col2 = st.columns(2)

        with col1:
            use_technical = st.checkbox("Gunakan Technical Indicators", value=True)
            use_ensemble = st.checkbox("Gunakan Ensemble Voting", value=True)

        with col2:
            feature_selection = st.checkbox("Gunakan Feature Selection", value=True)
            n_future_days = st.slider("Jumlah hari prediksi", 1, 14, 7)

        if st.button("🚀 MULAI ADVANCED PREDICTION", type="primary"):

            # STEP 1: TECHNICAL INDICATORS
            with st.spinner("Menghitung technical indicators..."):
                if use_technical:
                    df_enhanced = calculate_technical_indicators(df, price_col)
                    st.success(f"✅ Technical indicators ditambahkan: {len(df_enhanced.columns) - len(df.columns)} features baru")
                else:
                    df_enhanced = df.copy()
                    df_enhanced['day_num'] = range(len(df_enhanced))
                    df_enhanced['price_lag_1'] = df_enhanced[price_col].shift(1)
                    df_enhanced['price_lag_2'] = df_enhanced[price_col].shift(2)

            # Cleaning data
            df_clean = df_enhanced.dropna()
            st.info(f"📊 Data setelah cleaning: {len(df_clean)} baris")

            if len(df_clean) < 20:
                st.error("❌ Data terlalu sedikit untuk advanced analysis! Minimal 20 baris")
                st.stop()

            # STEP 2: PREPARE FEATURES
            exclude_cols = ['date_processed', price_col, date_col]
            feature_columns = [col for col in df_clean.columns
                             if col not in exclude_cols
                             and pd.api.types.is_numeric_dtype(df_clean[col])]

            X = df_clean[feature_columns]
            y = df_clean[price_col]

            # STEP 3: FEATURE SELECTION
            if feature_selection and len(feature_columns) > 10:
                with st.spinner("Melakukan feature selection..."):
                    selector = SelectFromModel(
                        RandomForestRegressor(n_estimators=100, random_state=42),
                        max_features=20
                    )
                    X_selected = selector.fit_transform(X, y)
                    selected_features = X.columns[selector.get_support()]
                    X = pd.DataFrame(X_selected, columns=selected_features, index=y.index)
                    st.success(f"✅ Feature selection: {len(selected_features)} features terpilih")

            # STEP 4: TRAIN MODELS
            st.subheader("🤖 Training Advanced Models")

            models = create_advanced_models()
            trained_models = {}
            results = {}

            progress_bar = st.progress(0)

            for i, (name, model) in enumerate(models.items()):
                # Time Series Cross Validation
                cv_score, cv_std = time_series_cross_validate(model, X, y)

                # Final training
                model.fit(X, y)
                trained_models[name] = model

                # Predictions untuk evaluation
                y_pred = model.predict(X)

                # Calculate metrics
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                mape = np.mean(np.abs((y - y_pred) / np.maximum(y, 1))) * 100

                results[name] = {
                    'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape,
                    'CV Score': cv_score, 'CV Std': cv_std
                }

                progress_bar.progress((i + 1) / len(models))
                st.success(f"✅ {name} - R²: {r2:.3f}, CV Score: {cv_score:.3f}")

            # STEP 5: ENSEMBLE VOTING
            if use_ensemble and len(trained_models) > 1:
                with st.spinner("Membuat ensemble voting..."):
                    ensemble_model = create_voting_ensemble(trained_models)
                    ensemble_model.fit(X, y)
                    ensemble_pred = ensemble_model.predict(X)

                    ensemble_mae = mean_absolute_error(y, ensemble_pred)
                    ensemble_r2 = r2_score(y, ensemble_pred)
                    ensemble_mape = np.mean(np.abs((y - ensemble_pred) / np.maximum(y, 1))) * 100

                    results['Ensemble Voting'] = {
                        'MAE': ensemble_mae, 'RMSE': np.sqrt(mean_squared_error(y, ensemble_pred)),
                        'R2': ensemble_r2, 'MAPE': ensemble_mape,
                        'CV Score': 0, 'CV Std': 0
                    }
                    trained_models['Ensemble Voting'] = ensemble_model

            # STEP 6: DISPLAY RESULTS
            st.subheader("📊 Hasil Advanced Training")

            results_df = pd.DataFrame(results).T
            results_df = results_df[['MAE', 'RMSE', 'R2', 'MAPE', 'CV Score', 'CV Std']]
            results_df.columns = ['MAE', 'RMSE', 'R²', 'MAPE (%)', 'CV R²', 'CV Std']

            # Sort by R² score
            results_df = results_df.sort_values('R²', ascending=False)

            col1, col2 = st.columns([2, 1])

            with col1:
                styled_df = results_df.style.format({
                    'MAE': '{:,.0f}', 'RMSE': '{:,.0f}',
                    'R²': '{:.3f}', 'MAPE (%)': '{:.2f}%',
                    'CV R²': '{:.3f}', 'CV Std': '{:.3f}'
                }).highlight_max(subset=['R²', 'CV R²'], color='lightgreen') \
                  .highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='lightgreen')

                st.dataframe(styled_df, use_container_width=True)

            with col2:
                best_model = results_df.index[0]
                best_r2 = results_df.iloc[0]['R²']
                best_mape = results_df.iloc[0]['MAPE (%)']

                st.metric("🎯 Model Terbaik", best_model)
                st.metric("📈 R² Score", f"{best_r2:.3f}")
                st.metric("🎯 Akurasi", f"{100 - best_mape:.1f}%")
                st.metric("📊 CV Consistency", f"{results_df.iloc[0]['CV R²']:.3f}")

            # STEP 7: FEATURE IMPORTANCE
            st.subheader("🔍 Feature Importance")

            importance_df = calculate_feature_importance(X, y)

            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                x=importance_df['importance'].head(15),
                y=importance_df['feature'].head(15),
                orientation='h',
                marker_color='lightblue'
            ))

            fig_importance.update_layout(
                title='15 Features Terpenting',
                xaxis_title='Importance',
                height=400
            )

            st.plotly_chart(fig_importance, use_container_width=True)

            # STEP 8: FUTURE PREDICTIONS
            st.subheader(f"🔮 Prediksi {n_future_days} Hari Ke Depan")

            last_data = df_clean.iloc[-1:].copy()
            current_price = last_data[price_col].iloc[0]

            all_predictions = {}

            for model_name, model in trained_models.items():
                predictions = []
                current_features = last_data[X.columns].copy()

                for day in range(n_future_days):
                    try:
                        # Predict next price
                        next_price = model.predict(current_features)[0]
                        predictions.append(max(next_price, 0.1))

                        # Update features untuk next prediction
                        if day < n_future_days - 1:
                            current_features = current_features.copy()

                            # Update lag features
                            for lag in [1, 2, 3, 5, 7, 10, 14]:
                                lag_col = f'price_lag_{lag}'
                                if lag_col in current_features.columns:
                                    if lag == 1:
                                        current_features[lag_col] = next_price
                                    else:
                                        # Shift other lags
                                        pass

                            # Update day number
                            if 'day_num' in current_features.columns:
                                current_features['day_num'] += 1

                            # Update rolling statistics (approximate)
                            for window in [3, 5, 7, 10, 14]:
                                mean_col = f'rolling_mean_{window}'
                                if mean_col in current_features.columns:
                                    current_features[mean_col] = (current_features[mean_col] * (window - 1) + next_price) / window

                    except Exception as e:
                        predictions.append(np.nan)

                all_predictions[model_name] = predictions

            # Create future dates
            if use_date:
                last_date = df_clean['date_processed'].max()
                future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(n_future_days)]
            else:
                last_day = df_clean['day_num'].max() if 'day_num' in df_clean.columns else len(df_clean)
                future_dates = [f"Hari {last_day + i + 1}" for i in range(n_future_days)]

            # Display best model predictions
            best_model_name = results_df.index[0]
            best_predictions = all_predictions[best_model_name]

            col1, col2 = st.columns(2)

            with col1:
                pred_df = pd.DataFrame({
                    'Tanggal': [d.strftime('%d/%m/%Y') if hasattr(d, 'strftime') else d for d in future_dates],
                    'Prediksi': [f"Rp {p:,.0f}" for p in best_predictions],
                    'Perubahan %': [f"{((p - current_price)/current_price*100):+.2f}%" for p in best_predictions]
                })
                st.dataframe(pred_df, use_container_width=True)

            with col2:
                avg_prediction = np.mean(best_predictions)
                total_change = ((best_predictions[-1] - current_price) / current_price) * 100
                trend = "📈 NAIK" if total_change > 0 else "📉 TURUN"

                st.metric("💰 Harga Terakhir", f"Rp {current_price:,.0f}")
                st.metric("📊 Rata-rata Prediksi", f"Rp {avg_prediction:,.0f}")
                st.metric("🎯 Trend Prediksi", trend, f"{total_change:+.2f}%")
                st.metric("🎲 Confidence Score", f"{(results_df.iloc[0]['R²'] * 100):.1f}%")

            # STEP 9: ADVANCED VISUALIZATION
            st.subheader("📈 Advanced Visualization")

            fig_advanced = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price Prediction', 'Technical Indicators'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Historical data
            show_days = min(60, len(df_clean))
            hist_dates = df_clean['date_processed'].iloc[-show_days:]
            hist_prices = df_clean[price_col].iloc[-show_days:]

            # Price prediction plot
            fig_advanced.add_trace(
                go.Scatter(
                    x=hist_dates, y=hist_prices,
                    mode='lines', name='Historical',
                    line=dict(color='blue', width=2)
                ), row=1, col=1
            )

            # Add predictions for best model
            fig_advanced.add_trace(
                go.Scatter(
                    x=future_dates, y=best_predictions,
                    mode='lines+markers', name=f'Prediksi ({best_model_name})',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8)
                ), row=1, col=1
            )

            # Add RSI if available
            if 'RSI_14' in df_clean.columns:
                rsi_data = df_clean['RSI_14'].iloc[-show_days:]
                fig_advanced.add_trace(
                    go.Scatter(
                        x=hist_dates, y=rsi_data,
                        mode='lines', name='RSI',
                        line=dict(color='purple', width=1)
                    ), row=2, col=1
                )

                # Add RSI reference lines
                fig_advanced.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig_advanced.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig_advanced.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_advanced, use_container_width=True)

            # STEP 10: DOWNLOAD RESULTS
            st.subheader("💾 Download Hasil Lengkap")

            # Prepare comprehensive results
            final_results = pd.DataFrame({
                'Tanggal': future_dates
            })

            for model_name, predictions in all_predictions.items():
                final_results[model_name] = predictions

            csv = final_results.to_csv(index=False)

            st.download_button(
                label="📥 Download Semua Prediksi",
                data=csv,
                file_name="advanced_predictions.csv",
                mime="text/csv"
            )

            st.success("🎉 Advanced Analysis Selesai!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Pastikan data Anda memiliki format yang benar dan cukup data historis.")

else:
    st.info("""
    ## 🚀 Advanced Stock Prediction

    **Fitur Advanced yang Ditambahkan:**

    ✅ **Technical Analysis:**
    - Moving Averages (SMA, EMA)
    - MACD, RSI, Bollinger Bands
    - Volatility measures (ATR, True Range)
    - Volume indicators

    ✅ **Advanced Machine Learning:**
    - Optimized hyperparameters
    - Ensemble Voting
    - Feature Selection
    - Time Series Cross Validation

    ✅ **Enhanced Features:**
    - Feature Importance Analysis
    - Advanced Visualization
    - Confidence Scoring
    - Multi-day predictions

    **Upload data CSV Anda untuk memulai analysis advanced!**
    """)

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Andriy - Advanced Prediction System")
