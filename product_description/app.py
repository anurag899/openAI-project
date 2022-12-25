import streamlit as st
from tasks import Task
from PIL import Image

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        filepath = uploaded_file.name
        st.image(Image.open(filepath))
        with col2:
            tk = Task()
            title = tk.getText(filepath)
            improve_title = tk.getGPTResponse('make product title more suitable',title)
            description = tk.getGPTResponse('generate product description points',improve_title)
            st.write("**Product Title**")
            st.write(improve_title)
            st.write("**Product Description**")
            st.write(description)
