import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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


        columns = df.columns.tolist()
        st.sidebar.header('Select Columns')
        x_feature = st.sidebar.selectbox('X-axis', options=columns)
        y_feature = st.sidebar.selectbox('Y-axis', options=columns)
        graph_type = st.sidebar.selectbox('Graph Type', options=['Scatter Plot', 'Line Plot', 'Bar Plot'])

        # Plot the data
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
