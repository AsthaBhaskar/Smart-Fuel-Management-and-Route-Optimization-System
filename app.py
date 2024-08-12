##app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Load dataset with st.cache_data
@st.cache_data
def load_data():
    dataset = pd.read_csv(r'C:\Users\sudhirb\fuel\tugboat_dataset (1).csv')
    return dataset

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_pred)], [min(y_test), max(y_pred)], linestyle='--', color='red')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    st.pyplot(plt)

# Function for simulated annealing route optimization
def simulated_annealing(points, initial_temp=1000, cooling_rate=0.003, stopping_temp=1e-8, num_iterations=1000):
    num_points = len(points)
    current_state = points[:]
    best_state = current_state[:]

    current_temp = initial_temp

    def predict_fuel_consumption(point1, point2):
        # Replace with your model's fuel consumption prediction logic
        # Example: Dummy calculation based on Euclidean distance
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def cost(route):
        total_cost = 0
        for i in range(num_points - 1):
            total_cost += predict_fuel_consumption(route[i], route[i + 1])
        return total_cost

    def get_neighbor(state):
        neighbor = state[:]
        idx1, idx2 = random.sample(range(num_points), 2)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    for iteration in range(num_iterations):
        next_state = get_neighbor(current_state)

        current_cost = cost(current_state)
        next_cost = cost(next_state)

        if next_cost < current_cost:
            current_state = next_state[:]
            if next_cost < cost(best_state):
                best_state = next_state[:]
        else:
            delta_cost = next_cost - current_cost
            accept_prob = math.exp(-delta_cost / current_temp)
            if random.random() < accept_prob:
                current_state = next_state[:]

        current_temp *= 1 - cooling_rate

        if current_temp < stopping_temp:
            break

    return best_state, cost(best_state)  # Return optimal route and its cost

# Main Streamlit app
def main():
    st.set_page_config(page_title='Smart Fuel Management and Route Optimization System', page_icon=':ship:', layout='wide')

    # Inject custom CSS for styling
    st.markdown("""
    <style>
    body {
        background-color: #f0f0f0; /* Light grey background */
        font-family: Arial, sans-serif; /* Font family */
    }
    .stTextArea {
        background-color: #ffffff; /* White background for text areas */
        border-color: #cccccc; /* Light grey border */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button background */
        color: white; /* White button text */
    }
    </style>
    """, unsafe_allow_html=True)

    st.title('Smart Fuel Management and Route Optimization System')


    # Load dataset
    dataset = load_data()

    # Define decoding mappings
    engine_type_decode = {1: 'Diesel-Electric', 0: 'Diesel'}
    tow_type_decode = {2: 'Ship', 0: 'Barge', 1: 'Other'}
    maintenance_history_decode = {1: 'Good', 0: 'Average', 2: 'Poor'}
    wind_direction_decode = {1: 'N', 2: 'NE', 0: 'E', 5: 'SE', 4: 'S', 6: 'SW', 7: 'W', 3: 'NW'}

    # Tabs navigation
    tabs = st.sidebar.radio('Navigation', ('Dataset Overview', 'Model Evaluation', 'Route Optimization'))

    if tabs == 'Dataset Overview':
        st.header('Dataset Overview')
        st.dataframe(dataset.head(10))  # Display first 10 rows

        # Data Visualization - Distribution Plots for Numerical Features
        st.subheader('Distribution Plots for Numerical Features')
        numeric_cols = ['Engine Power', 'Distance', 'Towing Speed', 'Tow Size', 'Current Speed', 'Fuel Consumption']
        for col in numeric_cols:
            st.write(f"Distribution plot of {col}")
            plt.figure(figsize=(10, 6))
            sns.histplot(dataset[col], kde=True)
            plt.title(col)
            st.pyplot(plt)

    elif tabs == 'Model Evaluation':
        st.header('Model Evaluation')

        # Perform feature selection and model training
        numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features.remove('Fuel Consumption')

        X_numerical = dataset[numerical_features]
        y_numerical = dataset['Fuel Consumption']

        select_k_best = SelectKBest(score_func=f_regression, k=10)
        select_k_best.fit(X_numerical, y_numerical)
        selected_features_indices = select_k_best.get_support(indices=True)
        selected_features_kbest = X_numerical.columns[selected_features_indices].tolist()

        X_selected = dataset[selected_features_kbest]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_numerical, test_size=0.2, random_state=42)

        gb_model = GradientBoostingRegressor()
        gb_model.fit(X_train, y_train)

        # Display evaluation metrics
        st.subheader('Evaluation Metrics')

        # User input for selected attributes
        st.sidebar.subheader('Select Attribute Values')

        user_input = {}
        for feature in selected_features_kbest:
            if feature == 'Engine Type':
                user_input[feature] = st.sidebar.selectbox(f'Select value for {feature}', list(engine_type_decode.values()))
            elif feature == 'Tow Type':
                user_input[feature] = st.sidebar.selectbox(f'Select value for {feature}', list(tow_type_decode.values()))
            elif feature == 'Maintenance History':
                user_input[feature] = st.sidebar.selectbox(f'Select value for {feature}', list(maintenance_history_decode.values()))
            elif feature == 'Wind Direction':
                user_input[feature] = st.sidebar.selectbox(f'Select value for {feature}', list(wind_direction_decode.values()))
            else:
                user_input[feature] = st.sidebar.number_input(f'Enter value for {feature}', value=float(X_train[feature].mean()))

        # Map selected values back to encoded values for prediction
        user_input_encoded = {}
        for feature, value in user_input.items():
            if feature == 'Engine Type':
                user_input_encoded[feature] = next(key for key, val in engine_type_decode.items() if val == value)
            elif feature == 'Tow Type':
                user_input_encoded[feature] = next(key for key, val in tow_type_decode.items() if val == value)
            elif feature == 'Maintenance History':
                user_input_encoded[feature] = next(key for key, val in maintenance_history_decode.items() if val == value)
            elif feature == 'Wind Direction':
                user_input_encoded[feature] = next(key for key, val in wind_direction_decode.items() if val == value)
            else:
                user_input_encoded[feature] = value

        # Ensure input_data has all necessary features
        input_data = pd.DataFrame([user_input_encoded], columns=selected_features_kbest)
        input_data = input_data.reindex(columns=X_selected.columns, fill_value=0)

        y_pred = gb_model.predict(input_data)

        # Store the random index in session state
        if 'random_idx' not in st.session_state:
            st.session_state.random_idx = random.randint(0, len(dataset) - 1)

        # Retrieve the actual fuel consumption from the dataset based on stored random index
        actual_value = dataset.iloc[st.session_state.random_idx]['Fuel Consumption']

        # Display actual and predicted values
        st.write('Actual Fuel Consumption:', actual_value)
        st.write('Predicted Fuel Consumption:', y_pred[0])

        # Display new evaluation metrics for the new prediction
        st.subheader('New Prediction Evaluation Metrics')
        y_test_new = np.append(y_test, actual_value)
        y_pred_new = np.append(gb_model.predict(X_test), y_pred[0])

        mse_new = mean_squared_error(y_test_new, y_pred_new)
        mae_new = mean_absolute_error(y_test_new, y_pred_new)
        r2_new = r2_score(y_test_new, y_pred_new)
        evs_new = explained_variance_score(y_test_new, y_pred_new)

        st.write('New Mean Squared Error (MSE):', mse_new)
        st.write('New Mean Absolute Error (MAE):', mae_new)
        st.write('New R-squared (R2):', r2_new)
        st.write('New Explained Variance Score (EVS):', evs_new)

        # Display actual vs predicted plot
        plot_actual_vs_predicted(y_test_new, y_pred_new)

    elif tabs == 'Route Optimization':
        st.header('Route Optimization')

        st.subheader('Adjust Coordinates')

        # User input for coordinates
        num_points = st.number_input('Number of Points', min_value=2, max_value=10, value=5, step=1)

        points = []
        for i in range(num_points):
            lat = st.number_input(f'Latitude {i+1}', value=float(random.uniform(0, 10)))
            lon = st.number_input(f'Longitude {i+1}', value=float(random.uniform(0, 10)))
            points.append((lat, lon))

        if st.button('Optimize Route'):
            optimal_route_sa, optimal_cost_sa = simulated_annealing(points)
            st.write('Optimal Route:', optimal_route_sa)
            st.write('Optimal Cost (Fuel Consumption):', optimal_cost_sa)

            # Plot optimal route
            def plot_route(route):
                latitudes = [point[0] for point in route]
                longitudes = [point[1] for point in route]

                plt.figure(figsize=(10, 6))
                plt.plot(longitudes, latitudes, marker='o', linestyle='-', color='b')

                # Plotting coordinates
                for i, txt in enumerate(route):
                    plt.annotate(f'{txt}', (longitudes[i], latitudes[i]), textcoords="offset points", xytext=(0, 10), ha='center')

                plt.title('Optimal Route with Least Fuel Consumption (Simulated Annealing)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True)
                st.pyplot(plt)

            plot_route(optimal_route_sa)

# Entry point of the app
if __name__ == '__main__':
    main()
