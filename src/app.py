import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import base64
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

# Function to create a download link for the dataframe
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    return href


# Function to handle missing values using KNNImputer
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

# Main function
def main():
    st.title('Data Analysis App 📊')
    st.write('Welcome to the Data Analysis App! This app is designed to help you analyze your dataset. You can upload your dataset, handle missing values, check for outliers, visualize your data, and perform clustering on your dataset with the help of a few simple clicks. Let\'s get started!!!')

    # Load the dataset
    st.sidebar.header('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""---""") 

    # Display the dataset
    if uploaded_file is not None:
        st.sidebar.write('File uploaded successfully!')
        st.sidebar.markdown("""---""") 
        df = pd.read_csv(uploaded_file)
        st.markdown("""---""") 
        st.subheader('Your Dataset: ')
        st.write(df.head())

        #Store the dataframe in session state
        if 'df' not in st.session_state:
            st.session_state.df = df

        st.markdown("""---""") 
        # Display the dataset shape
        st.subheader('Dataset Shape: ')
        st.write("The number of features in the dataset are: ", st.session_state.df.shape[1])
        st.write("The number of samples in the dataset are: ", st.session_state.df.shape[0])
        st.markdown("""---""") 

        # Display the dataset columns
        st.subheader('Dataset Columns: ')
        st.write('The dataset columns are: ', st.session_state.df.columns)
        st.markdown("""---""") 

        # Display the dataset summary
        st.subheader('Basic Dataset Summary: ')
        st.write('The dataset summary is: ', st.session_state.df.describe())
        st.markdown("""---""") 

        # Display columns with missing values
        st.subheader('Columns with Missing Values: ')
        missing_values = st.session_state.df.isnull().sum()
        st.write('The columns with missing values are: ')
        st.write(missing_values[missing_values>0])

        #Creating heatmap for the missing values
        st.write('Heatmap for missing values')
        df_corr = convert_categorical_to_numeric(st.session_state.df)
        fig = plt.figure()
        sns.heatmap(df_corr.isnull(), cbar=False)
        st.pyplot(fig)

        st.markdown("""---""") 

        

        #Plot and highlight Outliers in the user selected column
        st.sidebar.header('Outliers')
        st.sidebar.write('Outliers are the data points that are significantly different from the other data points in the dataset. The outliers can be detected by plotting the data points in the column selected by the user. The user can select the column and the method to detect the outliers.')
        column_outliers = st.sidebar.selectbox('Select Column', options=st.session_state.df.columns.tolist())
        method_outliers = st.sidebar.radio(f'How to detect outliers in {column_outliers}', options=['Z-Score', 'IQR'])
        if st.sidebar.button('Detect Outliers', key='detect_outliers'):
            st.subheader('Outliers')
            if method_outliers == 'Z-Score':
                z = np.abs((st.session_state.df[column_outliers] - st.session_state.df[column_outliers].mean()) / st.session_state.df[column_outliers].std())
                outliers = st.session_state.df[z > 3]
                st.write(f'The outliers in the dataset based on {column_outliers} using z-score method are: ', outliers)
                fig = plt.figure()
                sns.boxplot(x=column_outliers, data=st.session_state.df)
                st.pyplot(fig)
            elif method_outliers == 'IQR':
                Q1 = st.session_state.df[column_outliers].quantile(0.25)
                Q3 = st.session_state.df[column_outliers].quantile(0.75)
                IQR = Q3 - Q1
                outliers = st.session_state.df[(st.session_state.df[column_outliers] < (Q1 - 1.5 * IQR)) | (st.session_state.df[column_outliers] > (Q3 + 1.5 * IQR))]
                st.write(f'The outliers in the dataset based on {column_outliers} using IQR method are: ', outliers)
                fig = plt.figure()
                sns.boxplot(x=column_outliers, data=st.session_state.df)
                st.pyplot(fig)
            else:
                st.write('Invalid Method')
        st.sidebar.markdown("""---""") 


        #Creating Buttons to choose the method of handling missing values
        st.sidebar.header('Handle Missing Values')
        st.sidebar.write("Missing values can be dealt with by filling them with the mean, median, mode or by dropping the column. There is also an option to use KNN Imputer to handle missing values.")
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0].index.tolist()
        
        if missing_columns:
            column = st.sidebar.selectbox('Select Column', options=missing_columns)
            method = st.sidebar.radio(f'How to handle missing values in {column}', options=['Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Drop Column'])

            if 'column' not in st.session_state:
                st.session_state.column = column
            if 'method' not in st.session_state:
                st.session_state.method = method
            
    
            #st.write(st.session_state.df.head())
            if st.sidebar.button("Handle Missing Values" , key='handle_missing_values'):
                if method == 'Fill with Mean':
                    st.session_state.df[column].fillna(df[column].mean(), inplace=True)
                elif method == 'Fill with Median':
                    st.session_state.df[column].fillna(df[column].median(), inplace=True)
                elif method == 'Fill with Mode':
                    st.session_state.df[column].fillna(df[column].mode()[0], inplace=True)
                elif method == 'Drop Column':
                    st.session_state.df.drop(column, axis=1, inplace=True)
                else:
                    st.write('Invalid Method')

                # Display the number of missing values in the column after handling
                st.write(f'The data set after handling missing values is', st.session_state.df.head())
            st.sidebar.markdown("""---""") 


        # Use KNNImputer to handle missing values
        st.sidebar.header('Use KNN Imputer to Handle Missing Values')
        n_neighbors = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=10)
        if st.sidebar.button("Handle Missing Values with KNN Imputer", key='handle_missing_values_knn'):
            df = KNN_missing_values(st.session_state.df, n_neighbors)
            #st.write('The number of missing values in the dataset after handling are: ', st.session_state.df.isnull().sum().sum())
            st.write('The dataset after handling missing values is: ', st.session_state.df.head())
        st.sidebar.markdown("""---""") 
        

        if st.sidebar.button('Reset Data', key='reset_data'):
            st.session_state.df = df
            st.write('Data has been reset successfully!')
            st.write(st.session_state.df.head())
        st.sidebar.markdown("""---""") 


        st.sidebar.header('Data Visualization')
        #Drop down menu to select the categorical colums in the dataset
        st.sidebar.subheader('Select Categorical Columns')
        st.sidebar.write('Categories are the columns which have a fixed number of unique values. For example sex of a person can or type of a product can be termed as a category. The categories are plotted on the y-axis and the numerical columns are plotted on the x-axis. Only a bar graph can be plotted for categorical columns.')
        categorical_columns = st.sidebar.multiselect('Select Columns', options=st.session_state.df.columns.tolist())
        numerical_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        #Selecting the columns for plotting(Categorical)
        st.sidebar.subheader('Graph for Categorical Columns')
        st.sidebar.write('Select the x-axis and y-axis for plotting the bar graph. The x-axis consists of  numerical column and the y-axis consists of categorical columns selected.')
        x_feature_category = st.sidebar.selectbox('X-axis', options=numerical_columns , key='x_feature_category')
        y_feature_category = st.sidebar.selectbox('Y-axis(Categories)', options=categorical_columns , key='y_feature_category')

        #Bar Graph for Categorical Columns
        if st.sidebar.button('Plot Data' , key='plot_data_category'):
            st.subheader('Plotting Area')
            if x_feature_category and y_feature_category:
                fig = plt.figure()
                sns.barplot(x=x_feature_category, y=y_feature_category, data=st.session_state.df)
                st.pyplot(fig)


        #Selecting the columns for plotting(Numerical)
        st.sidebar.subheader('Graph for Numerical Columns')
        st.sidebar.write('Select the x-axis and y-axis for plotting the graph. The x-axis and y-axis consists of numerical columns selected. you can select the type of graph you want to plot.')
        x_feature_numeric = st.sidebar.selectbox('X-axis', options=numerical_columns , key='x_feature_numeric')
        y_feature_numeric = st.sidebar.selectbox('Y-axis', options=numerical_columns , key='y_feature_numeric')
        graph_type = st.sidebar.selectbox('Graph Type', options=['Scatter Plot', 'Line Plot', 'Bar Plot'])

        #Different Graphs for Numerical Columns
        if st.sidebar.button('Plot Data' , key='plot_data_numeric'):
            st.subheader('Plotting Area')
            if x_feature_numeric and y_feature_numeric:
                if graph_type == 'Scatter Plot':
                    st.write('Scatter Plot')
                    fig = plt.figure()
                    sns.scatterplot(x=x_feature_numeric, y=y_feature_numeric, data=st.session_state.df)
                    st.pyplot(fig)

                elif graph_type == 'Line Plot':
                    st.write('Line Plot')
                    fig = plt.figure()
                    sns.lineplot(x=x_feature_numeric, y=y_feature_numeric, data=st.session_state.df)
                    st.pyplot(fig)

                elif graph_type == 'Bar Plot':
                    st.write('Bar Plot')
                    fig = plt.figure()
                    sns.barplot(x=x_feature_numeric, y=y_feature_numeric, data=st.session_state.df)
                    st.pyplot(fig)

                else:
                    st.write('Invalid Graph Type')

        st.sidebar.markdown("""---""") 
        
        #Plotting clusters in a column
        st.sidebar.header('Clustering')
        st.sidebar.write('Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. The number of clusters to be formed can be selected by the user.')
        cluster_column = st.sidebar.selectbox('Select Column', options=numerical_columns)
        n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10)
        if st.sidebar.button('Plot Clusters', key='plot_clusters'):
            st.subheader('Clusters')
            if cluster_column:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters)
                st.session_state.df['Cluster'] = kmeans.fit_predict(st.session_state.df[[cluster_column]])
                fig = plt.figure()
                sns.scatterplot(x=cluster_column, y= cluster_column, hue='Cluster', data=st.session_state.df)
                st.pyplot(fig)
        
        #Display the entire dataset in a new tab
        st.sidebar.markdown("""---""") 
        if st.sidebar.button('View Entire Dataset', key='view_dataset'):
            st.write('Viewing the entire dataset')
            st.write(st.session_state.df)
            st.markdown(get_table_download_link(st.session_state.df), unsafe_allow_html=True)
        

    st.sidebar.markdown("""---""") 

    #    Button to view the entire dataset in a new tab






# Call the main function directly
if __name__ == "__main__":
    main()
