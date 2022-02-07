import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit_modal as modal
import streamlit.components.v1 as components

def app():
    # open_modal = st.button("Open")
    # if open_modal:
    #     modal.open()
    #
    #     if modal.is_open():
    #         with modal.container():
    #             st.write("Text goes here")
    #
    #             html_string = '''
    #             <h1>HTML string in RED</h1>
    #
    #             <script language="javascript">
    #             document.querySelector("h1").style.color = "red";
    #             </script>
    #             '''
    #             components.html(html_string)
    #
    #             st.write("Some fancy text")
    #             value = st.checkbox("Check me")
    #             st.write(f"Checkbox checked: {value}")
