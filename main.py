import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Adding a title to the sidebar
st.sidebar.title("Laptop Price Predictor WebApp")

# Displaying an image at the top of the sidebar
st.sidebar.image('laptop.png', use_column_width=True)


# Setting up the tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Data","Analysis"])

with tab1:
    # Application title
    st.markdown('<p class="big-font">Laptop Price Predictor</p>', unsafe_allow_html=True)

    # Columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox('Brand', df['Manufacturer'].unique())
        type = st.selectbox('Type', df['Category'].unique())
        cpu = st.selectbox('CPU', df['CPUbrand'].unique())
        gpu = st.selectbox('GPU', df['GPUbrand'].unique())
        os = st.selectbox('OS', df['OS'].unique())
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

    with col2:
        weight = st.number_input('Weight of the Laptop (kg)', format="%.2f")
        screen_size = st.number_input('Screen Size (inches)', format="%.2f")
        resolution = st.selectbox('Resolution',
                                  ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                                   '2560x1600', '2560x1440', '2304x1440'])
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
        ips = st.selectbox('IPS', ['No', 'Yes'])

    if st.button('Predict Price'):
        # Process inputs
        ppi = ((int(resolution.split('x')[0]) ** 2 + int(resolution.split('x')[1]) ** 2) ** 0.5) / screen_size
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
        predicted_price = np.exp(pipe.predict(query)[0])
        st.title(f"The predicted price of this configuration is £{predicted_price:.2f}")
        # Footer
    st.text("Developed by Cyrus Melroy Fernandes")


with tab2:
    st.header("Preview of Dataset")
    st.write("Here is a preview of the first few rows of the laptop dataset:")
    st.dataframe(df.head(20))

    st.text("Developed by Cyrus Melroy Fernandes")


with tab3:
    # vis 1
    st.title("Market Analysis")
    st.write("### Distribution of Laptop Prices (Pounds)")

    fig, ax = plt.subplots(figsize=(20, 10))  # Set the size of the figure
    sns.histplot(df['Price (Pounds)'], kde=True, color='teal', ax=ax, binwidth=100)
    ax.set_title('Distribution of Laptops by Price (Pounds)', fontsize=16)
    ax.set_xlabel('Price (Pounds)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    st.write(
        "Inference: The distribution of laptop prices is right skewed, with a few high-priced laptops. This may affect the convergence of our ML algorithm, suggesting normalization of the target variable.")


    # vis 2
    st.write("### Average Price of Laptops by Different Manufacturers")
    fig, ax = plt.subplots(figsize=(20, 10))  # Set the size of the figure
    sns.barplot(x = df['Manufacturer'], y = df['Price (Pounds)'], color='#ADAABF')
    ax.set_title('Average Price of Laptops by Different Manufacturers', fontsize=16)
    ax.set_xlabel('Manufacturer', fontsize=14)
    ax.set_ylabel('Price (Pounds)', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    # Inference
    st.write("""
        ### Inference
        Razer brand laptops have the highest average price. This is primarily because Razer specializes in high-configuration gaming laptops, which are typically more expensive due to their advanced graphics, faster processors, and higher-quality displays compared to standard laptops.
        """)


    # vis 3
    st.write("### Distribution of Laptops by RAM")
    fig, ax = plt.subplots(figsize=(20, 10))  # Set the size of the figure
    df['RAM'].value_counts().plot(kind='bar', color='mediumseagreen')
    ax.set_title('Distribution of Laptops by RAM', fontsize=16)
    ax.set_xlabel('RAM', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    # Inference
    # Inference
    st.write("""
    ### Inference
    8 GB RAM is the most commonly purchased configuration, followed by 4 GB and 16 GB. This trend suggests that 8 GB RAM hits a sweet spot for many users, balancing performance and cost effectively. It is sufficient for everyday tasks and some level of multitasking, making it a popular choice among general consumers.
    """)


    # vis4
    st.write("### Distribution of Laptops by RAM")
    fig, ax = plt.subplots(figsize=(20, 10))  # Set the size of the figure
    sns.barplot(x=df['OS'], y=df['Price (Pounds)'])
    ax.set_title('Distribution of Price for Laptops by OS', fontsize=16)
    ax.set_xlabel('OS', fontsize=14)
    ax.set_ylabel('Price (Pounds)', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    # Inference
    st.write("""
    ### Inference
    Mac OS exhibits the highest average price among the operating systems, followed by Windows and then other systems like Linux or Android. This can be attributed to the premium pricing strategy of Apple's hardware, which often integrates high-specification components and a strong ecosystem. Windows laptops come in a wider range of prices, reflecting their broad market reach and diverse user base. Other operating systems, typically found on more budget-friendly devices, generally present lower average prices.
    """)



    #vis5
    st.write("### Distribution of Laptops by Weight")

    fig, ax = plt.subplots(figsize=(20, 10))  # Set the size of the figure
    sns.histplot(df['Price (Pounds)'], kde=True, color='burlywood', ax=ax, binwidth=100)
    ax.set_title('Distribution of Laptops by Weight', fontsize=16)
    ax.set_xlabel('Weight', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    # Inference
    st.write("""
    ### Inference
    The distribution of laptop weights is shown to be mostly concentrated around the 2.0 to 2.5 kg range, 
    which suggests a common preference for moderately lightweight laptops. This might be attributed to a balance between the portability of lighter laptops and the robustness and 
    feature-richness of heavier models. Laptops lighter than 1.5 kg and heavier than 3 kg are less common, indicating less consumer 
    demand or practicality in these extremes—ultra-light laptops may compromise on screen size or power, while heavier ones may be less portable.
    """)



    st.text("Developed by Cyrus Melroy Fernandes")





