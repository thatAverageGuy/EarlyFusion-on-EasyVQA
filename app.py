import torch
from PIL import Image
from io import BytesIO
import streamlit as st
from model import EasyVQAEarlyFusion
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx2label = {0:'circle',
 1:'green',
 2:'red',
 3:'gray',
 4:'yes',
 5:'teal',
 6:'black',
 7:'rectangle',
 8:'yellow',
 9:'triangle',
 10:'brown',
 11:'blue',
 12:'no'}

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_language_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    text_encoder.to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False

    return tokenizer, text_encoder

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_vision_model():
    image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    
    return image_processor, image_encoder

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    model = EasyVQAEarlyFusion()
    model.load_state_dict(torch.load('./weights/final.model', map_location=device))
    model.to(device)
    return model

def process_text(question, tokenizer, text_encoder):
    text_inputs = tokenizer(question, return_tensors="pt")
    text_inputs = text_inputs.to(device)
    text_outputs = text_encoder(**text_inputs)
    text_embedding = text_outputs.pooler_output
    text_embedding = text_embedding.view(-1)
    text_embedding = text_embedding.detach()

    return text_embedding.view(1, -1)

def process_image(image, image_processor, image_encoder):
    image_inputs = image_processor(image, return_tensors="pt")
    image_inputs = image_inputs.to(device)
    image_outputs = image_encoder(**image_inputs)
    image_embedding = image_outputs.pooler_output
    image_embedding = image_embedding.view(-1)
    image_embedding = image_embedding.detach()

    return image_embedding.view(1, -1)

def predict(input, model):
    model.eval()
    with torch.no_grad():        
        output = model(**input)

    probs = round(torch.max(output.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()[0]*100, 4)
    output = output.argmax(-1)
    logit = output.detach().cpu().numpy()
    label = idx2label[logit[0]]

    return label, probs

if __name__=="__main__":

    st.title("Visual Question-Answering with Early Fusion Transformers on EasyVQA Dataset")
    with st.spinner('Models are being loaded.This could take some seconds...'):
        model=load_model()
        tokenizer, text_encoder = load_language_model()
        image_processor, image_encoder = load_vision_model()
    btn = None
    image_embedding = None
    text_embedding = None
    image = st.file_uploader("Upload an Image", type=['jpeg', 'jpg', 'png'], accept_multiple_files=False)
    
    if image:
        image_bytes = image.getvalue()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_embedding = process_image(img, image_processor=image_processor, image_encoder=image_encoder)
        
        st.image(img, caption="Uploaded Image")

        question = st.text_input('Enter input question: (Play with questions to see some funny results)', '')
        if question:
            text_embedding = process_text(question, tokenizer=tokenizer, text_encoder=text_encoder)
            btn = st.button("Answer")
    if btn:
        if image_embedding is not None and text_embedding is not None:
            with st.spinner("Running Prediction..."):
                input = {"image_emb":image_embedding, "text_emb":text_embedding}
                label, prob = predict(input, model)

            st.text(f"Answer: {label}\nConfidence: {prob}")