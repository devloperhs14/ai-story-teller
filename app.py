# import Few Libs

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#img-to-text (Hugging Face)
def img2text(path):
    img_to_text = pipeline("image-to-text", 
    model="Salesforce/blip-image-captioning-base")
    text = img_to_text(path)[0]['generated_text']
    print(text)
    return text

#llm - generates short story (use langchain)
def story_generator(scenario):
    template = """
    You are an expert kids story teller;
    You can generate short stories based on a simple narrative
    Your story should be more than 50 words.

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables = ["scenario"])
    story_llm = LLMChain(llm = OpenAI(
        model_name= 'gpt-3.5-turbo', temperature = 1), prompt=prompt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

#text-to-speech (Hugging Face)
def text2speech(msg):
    import requests
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {
         "inputs" : msg
    }
    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.flac','wb') as f:
        f.write(response.content)




def main():
    st.set_page_config(page_title = "AI story Teller", page_icon ="🤖")

    st.header("We turn images to story!")
    upload_file = st.file_uploader("Choose an image...", type = 'jpg')  #uploads image

    if upload_file is not None:
        print(upload_file)
        binary_data = upload_file.getvalue()
        
        # save image
        with open (upload_file.name, 'wb') as f:
            f.write(binary_data)
        st.image(upload_file, caption = "Image Uploaded", use_column_width = True) # display image

        scenario = img2text(upload_file.name) #text2image
        story = story_generator(scenario) # create a story
        text2speech(story) # convert generated text to audio

        # display scenario and story
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        # display the audio - people can listen
        st.audio("audio.flac")

# the main
if __name__ == "__main__":
    main()