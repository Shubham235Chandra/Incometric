import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def clustering():
    # Load and map data
    data = pd.read_csv('kmeans.csv')

    # Standard Scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, kmeans, clusters

def predict_cluster(age, work_experience, household_size, living_standards):

    # Set up the model and scaler
    model, scaler, kmeans, clusters = clustering()
    
    # Create the DataFrame using a dictionary
    input_data = pd.DataFrame({'age': age, 'work_experience': work_experience, 'household_size': household_size, 'living_standards': living_standards})

    # Define the mapping dictionary
    income_mapping = {"Low": 64207.0, "Medium": 77808.0, "High": 2485100.0}

    # Apply the mapping
    input_data['living_standards'] = input_data['living_standards'].map(income_mapping)

    X_scaled = scaler.fit_transform(input_data)
    predict = kmeans.predict(X_scaled)
    return predict[0]