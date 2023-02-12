import streamlit as st 
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import Feedback
import Model
import pandas as pd
from PIL import Image
import random


def main():
    st.set_page_config(page_title='HACKracker')
    # st.title('COVIDvisor')
    sidebarTitle = st.sidebar.title('HACKracker')
    
    menu = ['Home', 'Test', 'View Data', 'About']
    page = st.sidebar.selectbox(
        'Main Menu',
        ('Home', 'Test', 'View Data', 'About')
    )
    if page == "Test":
        st.title('Will you win or lose?')
        XGBClassifier, features = Model.train_data('Covid Dataset.csv')

        st.write(""" 
            ## Input your Project Title and we will provide you with a reliable feedback.
        """)
        

        #dataset_name = st.sidebar.selectbox(
            #'Select Illness to test for',
            #('COVID-19', 'Common Cold', 'Flu')
        #)

        #st.write(f"### Currently testing for {dataset_name} ")

        

        project_title = st.text_input( 'What is your Project Title?', )
        tagline = st.text_input( 'What is your tagline?', )
        description = st.text_input( 'Give a brief description of your project (in less than 50 words)', )
        media = st.selectbox( 'Did you provide a link to your stage pitch?', ('-----','Yes', 'No',))
        link = st.selectbox( 'Does your project contain a link to your GitHub account?', ('-----','Yes', 'No',))


        user_answers = [project_title,tagline,description,media,link,]
        if st.button("Get Checked"):
            
            page = "Home"
            st.success(f'Results submitted. Go to View Data page for detailed report')
            pos, neg = Model.test(XGBClassifier, Model.answer_conversion(user_answers))
            suggestedFeedback = Feedback.feedback(pos)
            st.write(f'Your topic has a {int(pos)} % of winning in the upcoming Hackathon')
            st.write(f'Feedback: {suggestedFeedback}')
            but = st.button("View Detail Results")
            # if but:
            #     page = 'View Data'
            # if st.button("Check your detailed report"):

    elif page == 'Home':
        st.title('HACKracker')
        
        st.write(""" 
            ## We will provide you with the chances of you winning the next hackathon! We monitor past,current & future trends to provide you with a detailed response.
            """)
            
        image = Image.open('hack.jpg')
        
        st.image(image, use_column_width=True)
        
        st.write('Our solution is a machine learning (web) application that provides users with an accurate statistic of their chances of winning a hackathon based on which category does their topic fit in. We make use of a trained prediction model which helps us to base our output based on trend analysis.')
        

    elif page == 'View Data':
        st.title('Your Result')
        col1,col2 = st.beta_columns(2)
        with col1:
            st.header('What should I do?')
            st.write('''
                Your topic chosen has a lot of interesting insights. 
                However, the title of the project has not been in demand in the recent years, 
                due to the saturation of similar ideologies from the specific industry chosen. 
                Winners from the years have worked in industries like Fintech, Education & Healthcare.
                
                Note: Choosing a topic from such industries does not guarantee you a win. It is a suggestion based
                      on the statistic displayed.
            ''')
            
            st.subheader('Key topic suggestion')
            st.write('**Fintech**')
            st.write('''
                Fintech is a growing & popular industry in the next decade. 
                Banking services, investment planning and reliable portfolios derivation 
                are approaching due to the fast-tracking technologies.
            ''')
            st.write('**Artificial Intelligence**')
            st.write('''
                An industry that would not be limited to talent and exposure in the near future.
                With the rapid advancements in hardware and software, AI technology has become more accessible, making it possible to develop and implement AI solutions in a variety of industries.
            ''')
            #st.write('**Contact with COVID-19 Patient**')
            #st.write('''
            #    COVID-19 can spread from an infected person's mouth or nose in small liquid particles when
            #    they cough, sneeze, speak, sing or breathe heavily.
            #''')


        with col2:
            st.header('Your winnning rate is **67%**')
            winrate = st.select_slider(
                '',
            options=['Low', 'Medium','High'],value='Medium')

            st.write('Your chances of winning are **', winrate, '**')

            st.button('Link to Suggested Topics')
            st.write('**Our Company Headquarters location is shown below: **')

            # insert map 
            data = [[22.336241357823056, 114.16659858285237]] 
            centerDF = pd.DataFrame(data, columns=['lat', 'lon'])
            st.map(centerDF)

            #st.write('Closest Test Centre: **Pak Tin Community Hall in Sham Shui Po**')

            st.write('**Address**: Student Residence City University of Hong Kong, Pak Tin Estate, Pak Tin Street, Shek Kip Mei, Sham Shui Po, Kowloon')
            st.write('**Contractor**: VSpaceCloud Limited')
            st.write('**Hotline**: 3611 1301 / 3611 1302')
            st.write('**Email**: vspacecloud@cityhack.hk')
    	
    else:
    	    st.title('VSpaceCloud')
    	    st.write('We are a company in the entertainment industry. VSpaceCLoud is a cutting-edge technology platform for the entertainment industry that provides a comprehensive suite of tools and data to help executives make informed decisions about the production and distribution of choices. Using artificial intelligence and machine learning algorithms, VSpaceCloud provides real-time analysis and predictions of performance, risks, and market trends.This helps companies optimize their diverse portfolios, improve marketing and distribution strategies, and ultimately maximize their returns on investment. With its innovative technology, VSpaceCloud is changing the way the entertainment industry operates, making it more efficient and data-driven.')


if __name__ == "__main__":
    main()
# st.write(f'Classifier = Random Forest')
# st.write(f'Answer = {user_answers}')

# pos, neg = Model.test(rand_forest, [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# # suggestedFeedback = Feedback.feedback(pos)

# st.write(f'You have : {pos} % of catching the Novel Coronavirus')

# st.write(f'Feedback: {suggestedFeedback}')
