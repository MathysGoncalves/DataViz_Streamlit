import pandas as pd
import datetime as dt
import numpy as np
import time

import streamlit as st
import streamlit.components.v1 as components 
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import pydeck as pdk


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

def st_log(func):
    def log_func(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        st.text("Log (%s): the function `%s` tooks %0.4f seconds" % (current_time, func.__name__, end))
        file1 = open("Logs.txt","a")
        file1.write("\nLog (%s): the function `%s` tooks %0.4f seconds" % (current_time, func.__name__, end))
        file1.close()
        return res

    return log_func


############ Data Prep. ############

@st.cache(allow_output_mutation=True)
def load_df1():
    df = pd.read_csv("uber-raw-data-apr14.csv")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df['Date'] = df['Date/Time'].dt.date

    df['Day'] = df['Date/Time'].dt.weekday
    df['WeekDay'] = df['Date/Time'].dt.day
    df['Hours'] = df['Date/Time'].dt.hour
    df['WeekDayName'] = df['Date/Time'].dt.day_name()
    df['count'] = 1
    df['lat'], df['lon'] = df['Lat'], df['Lon']

    return df

@st.cache(allow_output_mutation=True)
def load_df2():
    df = pd.read_csv("ny-trips-data.csv")
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['Hours_pickup'] = df['tpep_pickup_datetime'].dt.hour
    df['Hours_drop'] = df['tpep_dropoff_datetime'].dt.hour

    return df

########### Definition ###################

#@st.cache(hash_funcs={st.delta_generator.DeltaGenerator: lambda _: None})
def alt_barplot(col, df, aim):
    df_grouped = df.groupby(by=[aim]).sum()
    df_grouped = df_grouped.reset_index()

    bar = alt.Chart(df_grouped).mark_bar().encode( x=aim, y='count')#First plot
    rule = alt.Chart(df_grouped).mark_rule(color='red').encode(y='mean(count)')
        
    col.altair_chart(bar + rule)

def pw_heatmap(col, arg1, arg2, df, showscale):
    fig = px.imshow(df.groupby([arg1, arg2]).size().unstack())
    if showscale == False:
        fig.update_coloraxes(showscale=False)
    col.plotly_chart(fig)

def st_map(df):
    st.map(df.loc[(df['isin'] == True) & (df['in_date'] == True)])

def double_hist(col, df, arg1, arg2):
    fig = plt.hist([df[arg1], df[arg2]], bins = 24, rwidth=0.8, range=(0,24), label=[arg1, arg2])
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    plt.title('Frequency by Hours - NY')
    plt.legend(loc='upper right')
    col.pyplot(plt.show())

def plt_bar(col, df, arg1, arg2):
    df_grouped = df.groupby([arg1]).sum()
    fig = plt.bar(df_grouped.index, df_grouped[arg2])
    plt.xlabel('Hours of pickup')
    plt.ylabel('Passenger count')
    plt.title('Passenger count by Hours - NY')
    col.pyplot(plt.show())

def st_map2(col, df):
    col.map(df.loc[(df['isin'] == True) ])

def map_density(df, col):
   col.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
    ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
                data=df,
                get_position='[lon, lat]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
        ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
         ),
     ],
 ))

########### Components ################
_selectable_data_table = components.declare_component(
    "selectable_data_table", url="http://localhost:8501",
)


def selectable_data_table(data, key=None):
    return _selectable_data_table(data=data, default=[], key=key)


########### Visualisation ################
@st_log
def main():
    # First Dataset
    df_uber = load_df1()

    customer = df_uber.groupby(by=['lat', 'lon']).size()
    customer = pd.DataFrame(customer).sort_values(by=[0])

    # Second Dataset
    df_ny = load_df2()

    fare_sum = df_ny['fare_amount'].sum()
    tip_sum = df_ny['tip_amount'].sum()
    other_sum = (df_ny['total_amount'].sum() - (fare_sum + tip_sum))

    amount = [other_sum, fare_sum, tip_sum]

    st.markdown("<h1 style='text-align: center;'>Data Viz - Lab3</h1>", unsafe_allow_html=True)

    components.iframe("https://sp-ao.shortpixel.ai/client/to_webp,q_lossy,ret_img/https://www.24presse.com/ckfinder/userfiles/images/logo-uber.jpg")

    st.text("")
    st.write("In this lab, we will continue exploring the components available in Streamlit in order to make our web applications faster and cleaner. We will also see how we can easily deploy an application in the streamlit cloud.")
    st.write("""The lab parts
    During this lab, we will go through two main parts :
    - Understanding the concepts related to : python decorators, caching and memory optimization
    - Conceive and build your own streamlit component, all by yourself """)

    my_expander = st.expander("Summary")
    summary = my_expander.radio("Wich Dataset ?", ["Uber","NY Trips"]) #Choice of the dataset


    if summary == 'Uber':
        st.header("Uber Dataset ")

        #st.sidebar.title("Filter")
        #slider = st.sidebar.date_input('Select Date', [df_uber['Date/Time'].min(), df_uber['Date/Time'].max()])

        rows_uber = selectable_data_table(df_uber) #Bidirectional component
        if rows_uber:
            st.write("You have selected", rows_uber)

        col1, col2 = st.columns(2)
        col1.text("Head of the dataset")
        col1.write(df_uber.head(10))

        col2.text("Tail of the dataset")
        col2.write(df_uber.tail(10))

        st.text("")
        if st.checkbox('Show dataframe description'):
            col1.text("Dataset Description")
            col1.text("")
            col1.text("")
            col1.write(df_uber.describe())

            col2.text("Top Customer (Coord)")
            col2.write(customer)


        #Plot by Day
        #st.text("")
        #fig = plt.hist(df_uber['Day'], bins = 7, rwidth=0.8, range=(0.5,8))
        #plt.xlabel('Day')
        #plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        #plt.ylabel('Frequency')
        #plt.title('Frequency by Day - Uber - April 2014')
        #st.pyplot(plt.show())


        #Plot by Hours and WeekDay with mean
        st.text("")
        st.text("")
        st.markdown("<h3 style='text-align: center;'>Analysis by Time</h3>", unsafe_allow_html=True)
        st.text("")

        col1, col2, col3 = st.columns(3)

        alt_barplot(col1, df_uber, "Day") #Print both to Streamlit
        alt_barplot(col2, df_uber, "Hours")
        alt_barplot(col3, df_uber, "WeekDay")


        # Heatmap
        st.text("")
        st.markdown("<h3 style='text-align: center;'>Heatmap of the frequency</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        #fig, ax = plt.subplots(figsize=(4,3))
        #color = sns.color_palette("magma", as_cmap=True)
        #sns.plotting_context(font_scale=0.5)

        #sns.heatmap(df_uber.groupby(['WeekDay', 'Hours']).size().unstack(),ax=ax, linewidths=.2, cmap=color)
        #col1.write(fig)
        #sns.heatmap(df_uber.groupby(['Day', 'Hours']).size().unstack(),ax=ax, linewidths=.2, cmap=color)
        #col2.write(fig)

        pw_heatmap(col1, 'WeekDay', 'Hours', df_uber, False)
        pw_heatmap(col2, 'Day', 'Hours', df_uber, True)
        

        # Scatter
        st.markdown("<h3 style='text-align: center;'>Coordinate Analysis</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        tmp = df_uber

        col2.markdown('##') #Slider filtrer Date
        slider1, slider2 = col2.date_input('Select Date', [df_uber['Date'].min(), df_uber['Date'].max()])
        tmp['in_date'] = tmp['Date'].isin([slider1, slider2])

        options = col2.multiselect('Day to display :', pd.unique(df_uber['WeekDayName'])) # Choose Day of the week
        tmp['isin'] = tmp['WeekDayName'].isin(options)

        fig = px.scatter(df_uber.loc[(df_uber['isin'] == True) & (df_uber['in_date'] == True)], x="Lon", y="Lat", color="WeekDayName")
        col1.plotly_chart(fig)


        # Map
        st_map(df_uber)


    if summary == 'NY Trips':
        
        st.header("NY Trips Dataset ")

        rows_ny = selectable_data_table(df_ny) #Bidirectional component
        if rows_ny:
            st.write("You have selected", rows_ny)

        col1, col2 = st.columns(2)
        col1.text("Head of the dataset")
        col1.write(df_ny.head(10))

        col2.text("Tail of the dataset")
        col2.write(df_ny.tail(10))

        st.text("")
        if st.checkbox('Show dataframe description'):
            col1.text("Dataset Description")
            col1.text("")
            col1.text("")
            col1.write(df_ny.describe())

            fig = plt.pie(amount, labels=["Other", "Fare", "Tip"], autopct='%1.1f%%', explode=[0.05, 0.05, 0.05])
            plt.title('Amount')
            col2.pyplot(plt.show())


        # Time plot
        st.text("")
        st.text("")
        st.markdown("<h3 style='text-align: center;'>Analysis by Time</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        st.text("")
        df_ny['count'] = 1


        double_hist(col1, df_ny, 'Hours_pickup', 'Hours_drop')

        plt_bar(col2, df_ny, 'Hours_pickup', 'passenger_count')


        #Map Analisys
        st.markdown("<h3 style='text-align: center;'>Coordinate Analysis</h3>", unsafe_allow_html=True)

        coord_pick, coord_drop = df_ny, df_ny
        coord_pick['lon'], coord_pick['lat'] = coord_pick['pickup_longitude'], coord_pick['pickup_latitude']
        coord_pick = coord_pick.drop(['dropoff_longitude', 'dropoff_latitude'], axis=1)

        coord_drop['lon'], coord_drop['lat'] = coord_drop['dropoff_longitude'], coord_drop['dropoff_latitude']
        coord_drop = coord_drop.drop(['pickup_longitude', 'pickup_latitude'], axis=1)

        hour_selected = st.slider("Select hour of pickup", 0, 23)
        coord_pick['isin'] = coord_pick['Hours_pickup'] == hour_selected
        coord_drop['isin'] = coord_drop['Hours_drop'] == hour_selected

        col1, col2 = st.columns(2)

        st_map2(col1, coord_pick)
        st_map2(col2, coord_drop)

        
        col1.write("** Pickup New York City from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
        map_density(coord_pick, col1)

        col2.write("**Dropoff New York City**")
        map_density(coord_drop, col2)

    st.text("")
    st.text("")
    st.markdown("<h3 style='text-align: center;'>Other Dataset and Project available :</h3>", unsafe_allow_html=True)
    components.iframe("https://uber.github.io/#/", height=500, scrolling=True)


if __name__ == "__main__":
    main()