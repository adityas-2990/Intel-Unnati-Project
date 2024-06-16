import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# Function to convert boolean columns to numeric
def convert_booleans_to_numeric(df):
    df_encoded = df.copy()
    boolean_cols = df.select_dtypes(include=[bool]).columns
    df_encoded[boolean_cols] = df[boolean_cols].astype(int)
    return df_encoded

# Convert categorical columns to numeric
def convert_categorical_to_numeric(df):
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object' , 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df[col])
    return df_encoded

def KNN_missing_values(df , n_neighbors):
    df_imputed = df.copy()
    df_imputed = convert_booleans_to_numeric(df_imputed)
    df_imputed = convert_categorical_to_numeric(df_imputed)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = imputer.fit_transform(df_imputed)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

    #Copying the missing values back to the original dataset
    missing_columns = df.columns[df.isnull().any()]
    for column in missing_columns:
        df[column] = df_imputed[column]

    return df

def main():
    st.title('Data Visualization App')
    st.write('Welcome to the Data Visualization App!')

    # Load the dataset
    st.sidebar.header('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        st.sidebar.write('File uploaded successfully!')
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Display the dataset shape
        st.write("The number of features in the dataset are: ", df.shape[1])
        st.write("The number of samples in the dataset are: ", df.shape[0])

        # Display the dataset columns
        st.write('The dataset columns are: ', df.columns)

        # Display the dataset summary
        st.write('The dataset summary is: ', df.describe())

        # Display columns with missing values
        missing_values = df.isnull().sum()
        st.write('The columns with missing values are: ')
        st.write(missing_values[missing_values>0])


        #Creating heatmap for the missing values
        st.write('Heatmap for missing values')
        fig = plt.figure()
        sns.heatmap(df.isnull(), cbar=False)
        st.pyplot(fig)

        #Creating Buttons to choose the method of handling missing values
        st.sidebar.header('Handle Missing Values')
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0].index.tolist()
        if missing_columns:
            column = st.sidebar.selectbox('Select Column', options=missing_columns)
            method = st.sidebar.radio(f'How to handle missing values in {column}', options=['Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Drop Column'])

            if st.sidebar.button("Handle Missing Values" , key='handle_missing_values'):
                if method == 'Fill with Mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif method == 'Fill with Median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif method == 'Fill with Mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif method == 'Drop Column':
                    df.drop(column, axis=1, inplace=True)
                else:
                    st.write('Invalid Method')

                # Display the number of missing values in the column after handling
                st.write(f'The number of missing values in the "{column}" after handling are: ', df[column].isnull().sum())



        
        # User KNNImputer to handle missing values
        st.sidebar.header('Use KNN Imputer to Handle Missing Values')
        n_neighbors = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=10)
        if st.sidebar.button("Handle Missing Values with KNN Imputer", key='handle_missing_values_knn'):
            df = KNN_missing_values(df, n_neighbors)
            st.write('The number of missing values in the dataset after handling are: ', df.isnull().sum().sum())
            st.write('The dataset after handling missing values is: ', df.head())
        


        columns = df.columns.tolist()
        st.sidebar.header('Select Columns')
        x_feature = st.sidebar.selectbox('X-axis', options=columns)
        y_feature = st.sidebar.selectbox('Y-axis', options=columns)
        graph_type = st.sidebar.selectbox('Graph Type', options=['Scatter Plot', 'Line Plot', 'Bar Plot'])

        # Plot the data
        if st.sidebar.button('Plot Data' , key='plot_data'):
            if x_feature and y_feature:
                if graph_type == 'Scatter Plot':
                    st.write('Scatter Plot')
                    fig = plt.figure()
                    sns.scatterplot(x=x_feature, y=y_feature, data=df)
                    st.pyplot(fig)

                elif graph_type == 'Line Plot':
                    st.write('Line Plot')
                    fig = plt.figure()
                    sns.lineplot(x=x_feature, y=y_feature, data=df)
                    st.pyplot(fig)

                elif graph_type == 'Bar Plot':
                    st.write('Bar Plot')
                    fig = plt.figure()
                    sns.barplot(x=x_feature, y=y_feature, data=df)
                    st.pyplot(fig)

                else:
                    st.write('Invalid Graph Type')


# Call the main function directly
if __name__ == "__main__":
    main()
