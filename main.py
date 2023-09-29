import os

import streamlit as st
# from PIL import Image

# custom imports
from app.src.multipage import MultiPage
from app.src.constants import title

# import pages
from app.src.pages import project_introduction, global_analysis, individual_analysis, \
    preprocess_report, training_report


# Configuration page
st.set_page_config(page_title='WG analysis',
                   page_icon='app/images/wall-e.png',
                   layout="wide")

# Sidebar upside
# st.sidebar.image(np.array(Image.open('app/images/wall-e.png')),
#                 use_column_width=True)
st.sidebar.header(title)

# Create an instance of the app
app = MultiPage()

# Title of the main page
# display = np.array(Image.open('app/images/logo.png'))
# col1.image(display, use_column_width=True)
st.title(title)

# Add all your application here
app.add_page("Global Analysis", global_analysis.app)
app.add_page("Individual Analysis", individual_analysis.app)
app.add_page("Preprocess Report", preprocess_report.app)
app.add_page("Training Report", training_report.app)
app.add_page("Project Introduction", project_introduction.app)

# The main app
app.run()

# Sidebar downside
st.sidebar.markdown('Streamlit Dashboard to explore our analysis')
