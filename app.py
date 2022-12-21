import numpy as np
import pandas as pd
import streamlit as st
import json
import torch
from streamlit_option_menu import option_menu
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import speech_recognition as s 




with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home","Information"]
    )

   ################# Animation of the title
                    
st.markdown(
    """
<style>
.sidebar{
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)



st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://img.freepik.com/free-vector/seamless-gold-rhombus-grid-pattern-black-background_53876-97589.jpg?w=1060&t=st=1671168593~exp=1671169193~hmac=34116cdbd09587f0c6d4c289e5b48129239d316204877b7f5191f0c6e50d715d");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


if selected == "Home":

        col1,col2,col3,col4,col5,col6,col7,col8 = st.columns(8)




        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)


        Tbr = load_lottiefile("images/t.json")
        with col1:
            st_lottie(Tbr,
            loop=True,
            speed=0.28,
            key="hello")


        Obr = load_lottiefile("images/o.json")
        with col2:
            st_lottie(Obr,
            speed=0.28,
            key="O")


        Xbr = load_lottiefile("images/x.json")
        with col3:
            st_lottie(Xbr,
            speed=0.28,
            key="X")


        Ibr = load_lottiefile("images/i.json")
        with col4:
            st_lottie(Ibr,
            speed=0.28,
            key="I")



        Cbr = load_lottiefile("images/c.json")
        with col5:
            st_lottie(Cbr,
            speed=0.28,
            key="C")


        with col6:
            st_lottie(Ibr,
            speed=0.28,
            key="II")


        with col7:
            st_lottie(Tbr,
            speed=0.28,
            key="TT")


        Ybr = load_lottiefile("images/y.json")
        with col8:
            st_lottie(Ybr,
            speed=0.28,
            key="Y")





        ########################END OF ANIMATIONS



        st.markdown(f'<h1 style="color:#00ddb3;font-size:40px;border:2px solid white;margin-bottom:10px; text-align:center">{"Analysis of Text/Speech"}</h1>', unsafe_allow_html=True)

        tokenizer = AutoTokenizer.from_pretrained('unitary/toxic-bert')
        model = AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert')

        labels=('Toxic','Severe Toxic','Obscene/Sexual','Threat/Violent','Insulting','Identity Hate')

        st.markdown(f'<h1 style="color:#febf00;font-size:30px; text-shadow: 2px 2px 4px #ff0000; text-transform: uppercase;line-height: 1;">{"👉 Input Method"}</h1>', unsafe_allow_html=True)

        t1= st.radio("",
        ('Text','Speech'))

        st.markdown(
            """
            <style>
                .stProgress > div > div > div > div {
                    background-image: linear-gradient(to right, #5fde , #00cc42);
                }
            </style>""",
            unsafe_allow_html=True,
        )

        bar  = st.progress(0)
        
        if t1 == "Text":

                st.markdown(f'<h1 style="color:#febf00; text-shadow: 2px 2px 2px #ff0000;font-size:25px;">{"Enter Text : "}</h1>', unsafe_allow_html=True)

                text = st.text_area("👇")
                if st.button("Done"):
                    with st.spinner("Analysing Toxicity of the Text"):
                            time.sleep(1.5)
                            tokens = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)  
                            
                            result = model(tokens)    
                                
                            toxic_array=[]
                            
                            for i in range(6):
                                toxic_array.append((labels[i],round(float(result.logits[0][i]),3)))
                                bar.progress(50)
                                bar.progress(100) 
                                
                            max_= max(toxic_array[1:5], key= lambda tup:tup[1])
                            st.success("Result", icon="✅")
                            
                            if ((toxic_array[0][1]<=0) and (max_[1]<0)):
                                st.markdown(f'<h1 style="color:#00e500;text-shadow:1px 1px 1px orange;font-size:25px;">{"Not Toxic"}</h1>', unsafe_allow_html=True)

                            else:
                                st.markdown(f'<h1 style="color:#EC2001;text-shadow:1px 1px #36024c;font-size:30px;">⭕{max_[0]}</h1>', unsafe_allow_html=True)


                                count=0
                                for i in range(1,6):
                                    if toxic_array[i][1]>0:
                                        count+=1
                                if count>1:
                                    st.markdown(f'<h1 style="color:#febf00;font-size:28px;">{"Other Toxicity:"}</h1>', unsafe_allow_html=True)

                                    for i in range(1,6):
                                        if (toxic_array[i][1]>0) and (toxic_array[i]!=max_):
                                            st.markdown(f'<h1 style="color:#EC2001;text-shadow:1px 1px #36024c;font-size:25px;">⭕{toxic_array[i][0]}</h1>', unsafe_allow_html=True)




        else:
            st.markdown(f'<h1 style="color:#febf00;text-shadow: 2px 2px 3px #ff0000;font-size:23px;">{"🤏 Click on the Mic to Record"}</h1>', unsafe_allow_html=True)


            if st.button("🎙️"):

                sr=s.Recognizer()
                with s.Microphone() as m:
                    st.markdown(f'<h1 style="color:white; margin-bottom:5px;font-size:25px;">{"Speak Now..."}</h1>', unsafe_allow_html=True)

             
                    audio=sr.listen(m)
                    query=sr.recognize_google(audio,language='eng-in')
                    #st.image(query)
                    st.markdown(f'<h1 style="color:black;margin-bottom:7px;border-radius:10px;background-color:orange;padding:10px;border:2px solid white;font-size:20px;">{query}</h1>', unsafe_allow_html=True)

                    tokens = tokenizer.encode(query, return_tensors='pt', max_length=512, truncation=True)  
                    result = model(tokens)                             
                    toxic_array=[]
                    with st.spinner("Analysing Toxicity of the Text"):
                            time.sleep(1.5)                       
                            for i in range(6):
                                toxic_array.append((labels[i],round(float(result.logits[0][i]),3)))
                                bar.progress(50)
                                bar.progress(100)                    
                            max_= max(toxic_array[1:5], key= lambda tup:tup[1])
                            st.success('Result', icon="✅")

                            if ((toxic_array[0][1]<=0) and (max_[1]<0)):
                                                st.markdown(f'<h1 style="color:#00e500;font-size:25px;">{"Not Toxic"}</h1>', unsafe_allow_html=True)
                                                
                            else:
                                    st.markdown(f'<h1 style="color:#ffff;font-size:30px;">⭕{max_[0]}</h1>', unsafe_allow_html=True)
                                    count=0
                                    for i in range(1,6):
                                        if toxic_array[i][1]>0:
                                                count+=1
                                    if count>1:
                                        st.markdown(f'<h1 style="color:#EC2001;font-size:28px;">{"Other Toxicity:"}</h1>', unsafe_allow_html=True)

                                        for i in range(1,6):
                                            if (toxic_array[i][1]>0) and (toxic_array[i]!=max_):
                                                        st.markdown(f'<h1 style="color:#ffff;font-size:25px;">⭕{toxic_array[i][0]}</h1>', unsafe_allow_html=True)




if selected == "Information":
    st.markdown(f'<h1 style="color:#ffff;text-shadow:4px 4px 4px black; text-align:center;font-size:32px;">{"This Application uses BERT Model for Analysing Toxicity of Text/Speech"}</h1>', unsafe_allow_html=True)
       
    st.markdown(f'<h1 style="color:#febf00;margin-top:5rem;text-shadow:4px 4px 4px black; text-align:center;font-size:27px;">{"Made by :"}</h1>', unsafe_allow_html=True)
 
    st.markdown(f'<h1 style="color:#ffff;margin-top:0.3rem;text-shadow:4px 4px 4px black;text-align:center;font-size:25px;">{"🪶Aman Kaintura"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#ffff;margin-top:1rem;text-shadow:4px 4px 4px black; text-align:center;font-size:25px;">{"🪶Yuvraj Chakravarty"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#ffff;margin-top:1rem;text-shadow:4px 4px 4px black; text-align:center;font-size:25px;">{"🪶Bharat Kumar"}</h1>', unsafe_allow_html=True)

                                        


st.markdown("""
<style>
.css-1lsmgbg.egzxvld0
{
visibility: hidden;
}
</style>
""",unsafe_allow_html=True)

