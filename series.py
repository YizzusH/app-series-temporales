import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import holidays  # Librería para festivos fijos de Colombia

# Modelos estadísticos y Series Temporales
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet

# Configuración de la página
st.set_page_config(page_title="Forecasting & Time Series Analytics", layout="wide")

st.title("📈 Time Series Forecasting & Analytics Platform")
st.markdown("Plataforma interactiva profesional de series temporales con corrección robusta de índices semanales ISO, imputación natural diaria, intervalos de confianza blindados, Festivos de Colombia y Eventos Móviles.")


# ==========================================
# FUNCIONES PARA FESTIVOS Y EVENTOS MÓVILES
# ==========================================
def obtener_festivos_colombia(years):
    co_holidays = holidays.Colombia(years=years)
    list_holidays = []
    for date, name in co_holidays.items():
        list_holidays.append({
            'ds': pd.to_datetime(date),
            'holiday': name,
            'lower_window': 0,
            'upper_window': 0
        })
    return pd.DataFrame(list_holidays)

# Función para modelar demanda intermitente (Croston) con intervalo blindado
def croston_forecast(series, extra_periods):
    data = series.values
    d = np.asarray(data)
    cols = len(d)
    z = np.zeros(cols)
    p = np.zeros(cols)
    alpha = 0.1
    p_count = 1
    
    first_non_zero = np.argmax(d > 0)
    if d[first_non_zero] == 0:
        return np.zeros(extra_periods), np.zeros(extra_periods), np.zeros(extra_periods)
        
    z[0] = d[first_non_zero]
    p[0] = 1
    
    for i in range(1, cols):
        if d[i] > 0:
            z[i] = alpha * d[i] + (1 - alpha) * z[i-1]
            p[i] = alpha * p_count + (1 - alpha) * p[i-1]
            p_count = 1
        else:
            z[i] = z[i-1]
            p[i] = p[i-1]
            p_count += 1
            
    f = z[-1] / p[-1] if p[-1] != 0 else 0
    forecast_vals = np.full(extra_periods, f)
    
    std_err = np.std(d) if len(d) > 1 else 1.0
    ci_factor = np.minimum(1.96 * std_err * np.sqrt(np.arange(1, extra_periods + 1)), np.abs(forecast_vals) * 0.5)
    
    lower_vals = forecast_vals - ci_factor
    upper_vals = forecast_vals + ci_factor
    return forecast_vals, lower_vals, upper_vals


# ==========================================
# 1. CARGA DE DATOS Y CONFIGURACIÓN
# ==========================================
st.sidebar.header("1. Carga de Archivo")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📋 Vista Previa de los Datos")
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.dataframe(df.head(), use_container_width=True)
        date_col = st.selectbox("Selecciona la columna de Fecha:", df.columns)
        value_col = st.selectbox("Selecciona la columna Objetivo (Valor/Ventas):", [c for c in df.columns if c != date_col])

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df_clean = df[[date_col, value_col]].dropna().sort_values(by=date_col)
    
    min_date_found = df_clean[date_col].min().date()
    max_date_found = df_clean[date_col].max().date()

    # ==========================================
    # CONTROLES SIDEBAR: FECHAS Y FRECUENCIA
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.header("2. Rango de Fechas (Longitud de la Serie)")
    
    start_date, end_date = st.sidebar.date_input(
        "Seleccionar rango de fechas (DD/MM/AAAA):",
        value=(min_date_found, max_date_found),
        min_value=min_date_found,
        max_value=max_date_found,
        format="DD/MM/YYYY"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("3. Configuración del Pronóstico")
    
    freq_option = st.sidebar.selectbox(
        "Frecuencia de la serie:",
        options=["Días", "Semanas", "Meses", "Años"],
        index=2
    )
    
    freq_map = {"Días": "D", "Semanas": "W-MON", "Meses": "MS", "Años": "YS"}
    freq_code = freq_map[freq_option]

    if freq_option == "Días":
        date_format_str = '%d/%m/%Y'
        period_seasonal = 7
    elif freq_option == "Semanas":
        date_format_str = '%Y-S%V'
        period_seasonal = 52
    elif freq_option == "Meses":
        date_format_str = '%m/%Y'
        period_seasonal = 12
    else:
        date_format_str = '%Y'
        period_seasonal = 1
    # Etiquetas legibles: la semana usa el AÑO ISO (no el año calendario)
    def etiquetar(valores):
        idx = pd.DatetimeIndex(valores)
        if freq_option == "Semanas":
            iso = idx.isocalendar()
            return [f"{int(a)}-S{int(s):02d}" for a, s in zip(iso['year'], iso['week'])]
        return list(idx.strftime(date_format_str))
    horizon = st.sidebar.number_input(
        "Períodos a pronosticar hacia el futuro:",
        min_value=1, max_value=365, value=8, step=1
    )

    # ==========================================
    # 4. MÓDULO GENERAL DE EVENTOS MÓVILES / RANGOS
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.header("4. Eventos y Rangos Especiales")
    activar_eventos = st.sidebar.checkbox("Incluir eventos móviles o estacionales personalizados", value=False)
    
    df_moviles = []
    if activar_eventos:
        st.sidebar.info("Configura un evento especial o rango móvil de impacto:")
        nombre_evento = st.sidebar.text_input("Nombre del Evento (ej. Semana_Santa, Fin_de_ano):", "Semana_Santa")
        fecha_evento = st.sidebar.date_input("Fecha central del evento (DD/MM/AAAA):", value=min_date_found, format="DD/MM/YYYY")
        window_pre = st.sidebar.slider("Días previos de impacto (- días):", 0, 10, 2)
        window_post = st.sidebar.slider("Días posteriores de impacto (+ días):", 0, 10, 2)
        
        if st.sidebar.button("➕ Agregar Evento al Modelo"):
            if 'eventos_lista' not in st.session_state:
                st.session_state['eventos_lista'] = []
            st.session_state['eventos_lista'].append({
                'ds': pd.to_datetime(fecha_evento),
                'holiday': nombre_evento,
                'lower_window': -window_pre,
                'upper_window': window_post
            })
            st.sidebar.success(f"¡Evento '{nombre_evento}' agregado correctamente!")

        if 'eventos_lista' in st.session_state and len(st.session_state['eventos_lista']) > 0:
            st.sidebar.write("📌 **Eventos Activos:**")
            for idx, ev in enumerate(st.session_state['eventos_lista']):
                st.sidebar.text(f"- {ev['holiday']} ({ev['ds'].strftime('%d/%m/%Y')})")
            if st.sidebar.button("🗑️ Limpiar Eventos"):
                st.session_state['eventos_lista'] = []
                st.rerun()

        if 'eventos_lista' in st.session_state:
            df_moviles = st.session_state['eventos_lista']

    # Filtrar serie por el rango de fechas seleccionado
    mask = (df_clean[date_col].dt.date >= start_date) & (df_clean[date_col].dt.date <= end_date)
    ts_filtered = df_clean.loc[mask].set_index(date_col)

    # Procesamiento con imputación natural SOLO para Días; Suma acumulativa y ordenamiento ISO robusto para Semanas
    if freq_option == "Días":
        ts_data = ts_filtered.asfreq('D').ffill().bfill().sort_index()
        imputation_msg = "Sistemática Diaria: Días faltantes imputados mediante continuidad natural (sin ceros artificiales)."
    elif freq_option == "Semanas":
        # Semanas de lunes a domingo, etiquetadas con el lunes de inicio
        ts_data = ts_filtered.resample('W-MON', label='left', closed='left').sum()
        # Quitar semanas incompletas en los extremos (evita caídas artificiales al sumar)
        _ini, _fin = ts_filtered.index.min(), ts_filtered.index.max()
        if len(ts_data) > 2 and ts_data.index[0] < _ini:
            ts_data = ts_data.iloc[1:]
        if len(ts_data) > 2 and ts_data.index[-1] + pd.Timedelta(days=6) > _fin:
            ts_data = ts_data.iloc[:-1]
        imputation_msg = "Sistemática Semanal (lunes a domingo): cada semana se etiqueta con su lunes de inicio." 
    elif freq_option == "Años":
        ts_data = ts_filtered.resample('YS').sum().sort_index()
        imputation_msg = "Sistemática Anual: Valores consolidados por suma anual."
    else:
        ts_data = ts_filtered.resample(freq_code).sum().sort_index()
        imputation_msg = f"Sistemática Agrupada ({freq_option}): Valores consolidados por suma."

    años_serie = list(range(start_date.year, end_date.year + 2))
    df_holidays_co = obtener_festivos_colombia(años_serie)
    
    if len(df_moviles) > 0:
        df_moviles_pd = pd.DataFrame(df_moviles)
        df_holidays_co = pd.concat([df_holidays_co, df_moviles_pd], ignore_index=True)

    # ==========================================
    # 2. DIAGNÓSTICO Y ESTADÍSTICA DESCRIPTIVA
    # ==========================================
    st.markdown("---")
    st.header("📊 Estadística Descriptiva y Diagnóstico")
    st.info(f"ℹ️ **Lógica Aplicada:** {imputation_msg}")
    
    zero_count = (ts_data[value_col] == 0).sum()
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Períodos Analizados", len(ts_data))
    col_s2.metric("Períodos con Venta Cero", zero_count)
    col_s3.metric("Festivos Registrados", len(df_holidays_co))
    col_s4.metric("Promedio por Período", f"{ts_data[value_col].mean():.2f}")
    col_s5.metric("Venta Máxima", f"{ts_data[value_col].max():.2f}")

    # ==========================================
    # 3. MÓDULO DE ANÁLISIS ESTADÍSTICO Y GRÁFICOS ESTACIONALES
    # ==========================================
    st.markdown("---")
    st.header("🔬 Análisis Estadístico Avanzado y Gráficos Estacionales")

    # A1. Gráfico Estacional Superpuesto (Con manejo ISO seguro para Semanas)
    if freq_option != "Años":
        try:
            df_seasonal_plot = ts_data.copy()
            if freq_option == "Semanas":
                iso_c = df_seasonal_plot.index.isocalendar()
                df_seasonal_plot['Year'] = iso_c.year
                df_seasonal_plot['Period'] = iso_c.week
            else:
                df_seasonal_plot['Year'] = df_seasonal_plot.index.year
                df_seasonal_plot['Period'] = df_seasonal_plot.index.month if freq_option in ["Meses", "Días"] else 1
            
            pivot_seasonal = df_seasonal_plot.pivot_table(index='Period', columns='Year', values=value_col).sort_index()
            
            fig_seasonal = go.Figure()
            years_list = sorted(pivot_seasonal.columns)
            max_year = years_list[-1] if len(years_list) > 0 else None
            total_prev = max(1, len(years_list) - 1)
            
            for idx, year in enumerate(years_list):
                if year == max_year:
                    line_color = '#1d4ed8'  # Azul oscuro destacado
                    line_width = 3
                    opacity = 1.0
                else:
                    shade = int(200 - (idx / total_prev) * 90) if total_prev > 0 else 150
                    line_color = f'rgb({shade}, {shade}, {shade})'
                    line_width = 1.5
                    opacity = 0.7
                    
                fig_seasonal.add_trace(go.Scatter(
                    x=pivot_seasonal.index, 
                    y=pivot_seasonal[year], 
                    mode='lines', 
                    name=str(year),
                    line=dict(color=line_color, width=line_width),
                    opacity=opacity
                ))
                
            fig_seasonal.update_layout(
                title=f"Gráfico Estacional Superpuesto ({freq_option} por Ciclo - Destacando {max_year})",
                xaxis_title="Período del Ciclo",
                yaxis_title="Valor / Ventas",
                height=450,
                template="plotly_white"
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
        except Exception as e:
            st.info(f"Nota en gráfico estacional: {e}")
    else:
        n_anios = len(ts_data)
        min_requerido = 6
        if n_anios < min_requerido:
            st.warning(f"⚠️ **Alerta de Suficiencia Histórica (Frecuencia Anual):** Actualmente la serie cuenta con **{n_anios} años** registrados. Se requieren **al menos {min_requerido} años** (idealmente de 6 a 10) para visualizar patrones estacionales o tendencias anuales robustas sin comprometer la precisión estadística.")
        else:
            st.info(f"ℹ️ Frecuencia anual activa con {n_anios} años registrados. El gráfico estacional superpuesto se omite al ser una vista de agregación anual.")

    # B. Descomposición Clásica (Aditiva)
    try:
        if len(ts_data) >= 2 * period_seasonal and period_seasonal > 1:
            decomposition = seasonal_decompose(ts_data[value_col].sort_index(), model='additive', period=period_seasonal)
            
            fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                       subplot_titles=('Serie Original', 'Tendencia', 'Estacionalidad', 'Residuo (Aleatorio)'))
            
            x_dates = ts_data.index
            fig_decomp.add_trace(go.Scatter(x=x_dates, y=decomposition.observed, mode='lines', name='Observada'), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=x_dates, y=decomposition.trend, mode='lines', name='Tendencia'), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=x_dates, y=decomposition.seasonal, mode='lines', name='Estacionalidad'), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=x_dates, y=decomposition.resid, mode='markers', name='Residuo'), row=4, col=1)
            
            fig_decomp.update_layout(height=700, title_text="Descomposición Clásica Aditiva de la Serie Temporal", showlegend=False, template="plotly_white")
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.info("ℹ️ La serie es demasiado corta para la descomposición estacional clásica (se requieren al menos 2 ciclos completos).")
    except:
        pass

    # C. Pruebas de Estacionariedad y ACF
    col_stat1, col_stat2 = st.columns(2)
    series_vals = ts_data[value_col].dropna()

    with col_stat1:
        st.subheader("🧪 Pruebas de Estacionariedad")
        if len(series_vals) > 5:
            adf_result = adfuller(series_vals)
            st.markdown(f"**Test Dickey-Fuller Aumentado (ADF):**")
            st.write(f"- Estadístico ADF: `{adf_result[0]:.4f}`")
            st.write(f"- p-valor: `{adf_result[1]:.4f}`")
            if adf_result[1] < 0.05:
                st.success("✅ **ADF:** Serie **Estacionaria**.")
            else:
                st.warning("⚠️ **ADF:** Serie **No Estacionaria**.")
    with col_stat2:
        st.subheader("📉 Autocorrelación (ACF)")
        try:
            nlags_val = min(10, len(series_vals) // 2 - 1)
            if nlags_val > 1:
                lag_acf = acf(series_vals, nlags=nlags_val)
                fig_ac = go.Figure()
                fig_ac.add_trace(go.Bar(y=lag_acf, name='ACF'))
                fig_ac.update_layout(title="Gráfico de Autocorrelación (ACF)", xaxis_title="Rezaga (Lag)", yaxis_title="Coeficiente", height=300, template="plotly_white")
                st.plotly_chart(fig_ac, use_container_width=True)
        except:
            pass

    # ==========================================
    # 4. DETECCIÓN DE ANOMALÍAS
    # ==========================================
    st.markdown("---")
    st.header("🚨 Detección de Anomalías (Isolation Forest)")
    
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    ts_data['anomalia'] = iso_model.fit_predict(ts_data[[value_col]])
    
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=ts_data.index, 
        y=ts_data[value_col], 
        mode='lines', 
        name='Serie Ajustada'
    ))
    anom_points = ts_data[ts_data['anomalia'] == -1]
    fig_anom.add_trace(go.Scatter(
        x=anom_points.index, 
        y=anom_points[value_col], 
        mode='markers', 
        name='Anomalía', 
        marker=dict(color='red', size=8)
    ))
    fig_anom.update_layout(title="Anomalías Detectadas en la Serie", xaxis_title=f"Fecha ({freq_option})", yaxis_title="Valor/Ventas", template="plotly_white")
    st.plotly_chart(fig_anom, use_container_width=True)

    # ==========================================
    # 5. EVALUACIÓN Y PRONÓSTICO (SPLIT 80/20)
    # ==========================================
    st.markdown("---")
    st.header(f"⚙️ Evaluación de Modelos y Proyección Futura ({horizon} {freq_option})")

    train_size = max(3, int(len(ts_data) * 0.8))
    train, test = ts_data.iloc[:train_size], ts_data.iloc[train_size:]

    st.info(f"Histórico de Entrenamiento (80%): **{train.index.min().strftime(date_format_str)}** a **{train.index.max().strftime(date_format_str)}** ({len(train)} obs)")
    if len(test) > 0:
        st.info(f"Muestra Test de Validación (20%): **{test.index.min().strftime(date_format_str)}** a **{test.index.max().strftime(date_format_str)}** ({len(test)} obs)")

    val_forecasts = {}
    val_forecasts['Naive'] = np.repeat(train[value_col].iloc[-1], len(test)) if len(test) > 0 else np.array([train[value_col].iloc[-1]])
    
    if len(test) > 0:
        try:
            hw_model = ExponentialSmoothing(train[value_col], trend='add', seasonal=None).fit()
            val_forecasts['Holt-Winters'] = hw_model.forecast(len(test))
        except:
            val_forecasts['Holt-Winters'] = val_forecasts['Naive']

        try:
            ari_model = SARIMAX(train[value_col], order=(1, 1, 1)).fit(disp=False)
            val_forecasts['ARIMA'] = ari_model.forecast(len(test))
        except:
            val_forecasts['ARIMA'] = val_forecasts['Naive']

        try:
            croston_val, _, _ = croston_forecast(train[value_col], len(test))
            val_forecasts['Croston'] = croston_val
        except:
            val_forecasts['Croston'] = val_forecasts['Naive']

        try:
            prophet_df = train.reset_index().rename(columns={date_col: 'ds', value_col: 'y'})
            m_val = Prophet(
                yearly_seasonality=(freq_option != "Años"),
                weekly_seasonality=(freq_option == "Días"),
                daily_seasonality=False,
                holidays=df_holidays_co if freq_option != "Años" else None
            )
            m_val.fit(prophet_df)
            future_val = pd.DataFrame({'ds': test.index})
            p_val = m_val.predict(future_val)
            val_forecasts['Prophet (Con Eventos)'] = p_val['yhat'].values
        except:
            val_forecasts['Prophet (Con Eventos)'] = val_forecasts['Naive']

        # Métricas de Error
        metrics_list = []
        for model_name, f_val in val_forecasts.items():
            rmse = np.sqrt(mean_squared_error(test[value_col], f_val))
            mae = mean_absolute_error(test[value_col], f_val)
            y_true = test[value_col].values
            non_zero = y_true != 0
            mape = np.mean(np.abs((y_true[non_zero] - f_val[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else 0.0
            metrics_list.append({"Modelo": model_name, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})

        df_metrics = pd.DataFrame(metrics_list).sort_values(by="RMSE")
        best_model_name = df_metrics.iloc[0]["Modelo"]

        st.subheader("📊 Métricas de Error (Muestra de Prueba Test)")
        st.dataframe(df_metrics.style.highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE (%)"], color="#dcfce7"), use_container_width=True)
        st.success(f"🏆 **Modelo Recomendado:** **{best_model_name}** según menor RMSE.")
    else:
        best_model_name = 'Holt-Winters'
        df_metrics = pd.DataFrame([{"Modelo": best_model_name, "RMSE": 0, "MAE": 0, "MAPE (%)": 0}])
        st.info("ℹ️ La serie es muy corta para realizar partición 80/20 de validación; utilizando modelo estándar por defecto.")

    # ==========================================
    # PRONÓSTICO FUTURO CON INTERVALOS BLINDADOS
    # ==========================================
    last_date = ts_data.index.max()
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_code)[1:]

    future_forecast = []
    future_intervals = None

    if 'Prophet' in best_model_name:
        try:
            full_df = ts_data.reset_index().rename(columns={date_col: 'ds', value_col: 'y'})
            m_full = Prophet(
                yearly_seasonality=(freq_option != "Años"),
                weekly_seasonality=(freq_option == "Días"),
                daily_seasonality=False,
                holidays=df_holidays_co if freq_option != "Años" else None
            )
            m_full.fit(full_df)
            future_df = pd.DataFrame({'ds': future_dates})
            p_full = m_full.predict(future_df)
            future_forecast = p_full['yhat'].values
            future_intervals = {
                'lower': p_full['yhat_lower'].values,
                'upper': p_full['yhat_upper'].values
            }
        except:
            best_model_name = 'Holt-Winters'

    if best_model_name == 'ARIMA':
        try:
            ari_full = SARIMAX(ts_data[value_col], order=(1, 1, 1)).fit(disp=False)
            forecast_obj = ari_full.get_forecast(steps=horizon)
            future_forecast = forecast_obj.predicted_mean.values
            ci = forecast_obj.conf_int(alpha=0.05)
            future_intervals = {
                'lower': ci.iloc[:, 0].values,
                'upper': ci.iloc[:, 1].values
            }
        except:
            best_model_name = 'Holt-Winters'

    if best_model_name == 'Holt-Winters':
        try:
            hw_full = ExponentialSmoothing(ts_data[value_col], trend='add', seasonal=None).fit()
            future_forecast = hw_full.forecast(horizon).values
            residuals = hw_full.fittedvalues - ts_data[value_col]
            std_resid = np.std(residuals) if len(residuals) > 1 else 1.0
            ci_factor = np.minimum(1.96 * std_resid * np.sqrt(np.arange(1, horizon + 1)), np.abs(future_forecast) * 0.4)
            future_intervals = {
                'lower': future_forecast - ci_factor,
                'upper': future_forecast + ci_factor
            }
        except:
            future_forecast = np.repeat(ts_data[value_col].iloc[-1], horizon)
            std_naive = np.std(ts_data[value_col]) if len(ts_data) > 1 else 1.0
            ci_factor = 1.96 * std_naive * np.sqrt(np.arange(1, horizon + 1))
            future_intervals = {
                'lower': future_forecast - ci_factor,
                'upper': future_forecast + ci_factor
            }
    elif best_model_name == 'Croston':
        future_forecast, lower_c, upper_c = croston_forecast(ts_data[value_col], horizon)
        future_intervals = {
            'lower': lower_c,
            'upper': upper_c
        }

    # ==========================================
    # 6. GRÁFICA INTERACTIVA DEL PRONÓSTICO (EMPALME + INTERVALOS BLINDADOS)
    # ==========================================
    st.markdown("---")
    st.header(f"📈 Proyección Futura para los Próximos {horizon} {freq_option}")

    fig_future = go.Figure()
    
    fig_future.add_trace(go.Scatter(
        x=ts_data.index,
        y=ts_data[value_col],
        mode='lines',
        name='Histórica Selección'
    ))
    
    bridge_dates = [ts_data.index[-1]] + list(future_dates)
    bridge_forecast = [ts_data[value_col].iloc[-1]] + list(future_forecast)

    fig_future.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_forecast,
        mode='lines+markers',
        name=f'Pronóstico Futuro ({best_model_name})',
        line=dict(color='green', width=2, dash='dash')
    ))

    if future_intervals is not None:
        bridge_upper = [ts_data[value_col].iloc[-1]] + list(future_intervals['upper'])
        bridge_lower = [ts_data[value_col].iloc[-1]] + list(future_intervals['lower'])
        
        fig_future.add_trace(go.Scatter(
            x=bridge_dates + bridge_dates[::-1],
            y=bridge_upper + bridge_lower[::-1],
            fill='toself',
            fillcolor='rgba(22, 163, 74, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confianza (95%)'
        ))

    fig_future.update_layout(
        title=f"Proyección a {horizon} {freq_option} con el Modelo {best_model_name}",
        xaxis_title=f"Fecha ({freq_option})",
        yaxis_title="Valor / Ventas",
        template="plotly_white"
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # ==========================================
    # 7. DESCARGA DEL REPORTE EXCEL
    # ==========================================
    st.markdown("---")
    st.header("📥 Descargar Reporte en Excel")
    
    df_export_future = pd.DataFrame({
        f'Fecha_Proyectada ({freq_option})': future_dates.strftime(date_format_str),
        'Pronostico': future_forecast
    })
    
    if future_intervals is not None:
        df_export_future['Limite_Inferior'] = future_intervals['lower']
        df_export_future['Limite_Superior'] = future_intervals['upper']

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_metrics.to_excel(writer, sheet_name='Metricas_Evaluacion', index=False)
        df_export_future.to_excel(writer, sheet_name=f'Pronostico_{horizon}_Eventos', index=False)
        df_holidays_co.to_excel(writer, sheet_name='Eventos_Y_Festivos', index=False)
    
    st.download_button(
        label=f"↓ Descargar Pronóstico ({horizon} {freq_option}) en Excel",
        data=output.getvalue(),
        file_name=f"Pronostico_{horizon}_{freq_option}_{best_model_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👋 Por favor sube un archivo CSV o Excel en la barra lateral para comenzar.")
