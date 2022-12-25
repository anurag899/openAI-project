import openai
import easyocr
import streamlit as st
openai.api_key = <your-api-key>

reader = easyocr.Reader(['en'], gpu = False)


class Task:
    def getText(self,filepath):
        result = reader.readtext(filepath)
        return " ".join([i[1] for i in result])

    def getGPTResponse(self,task,text):
        response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"{task} for - {text}",
                temperature=0,
                max_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
        return response['choices'][0]['text']


    

    
