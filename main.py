from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import joblib
import pandas as pd
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from helper import download_hugging_face_embeddings
from transformers import pipeline
from prompt import prompt
from fastapi import Request


# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI and logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Define request models
class PredictionInput(BaseModel):
    hours_studied: float
    previous_scores: float
    extracurricular: str  # "Yes" or "No"
    sleep_hours: float
    question_papers: int
    feedback: str = "I'm stressed and need better planning."  # optional

class ChatInput(BaseModel):
    msg: str
    user_id: str  # Assuming you will pass user ID from Django
    predicted_score: float
    study_hours: float
    motivation_level: str
    preferred_learning_style: str



# Load models
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="students-score-padi", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = OpenAI(temperature=0.4, max_tokens=500)
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Create the document chain and retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Load the trained model
try:
    model = joblib.load("exam_score_predictor.pkl")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Student Performance App!"}

# Prepare input for prediction
def prepare_input(data: PredictionInput):
    return pd.DataFrame([{
        'Hours Studied': data.hours_studied,
        'Previous Scores': data.previous_scores,
        'Extracurricular Activities': data.extracurricular,
        'Sleep Hours': data.sleep_hours,
        'Sample Question Papers Practiced': data.question_papers
    }])

# Predict score endpoint
@app.post("/predict")
def predict_score(input_data: PredictionInput):
    df = prepare_input(input_data)
    predicted_score = model.predict(df)[0]
    return {"predicted_score": round(predicted_score, 2)}


@app.post("/chatbot-advice")
async def get_chatbot_advice(input_data: ChatInput, request: Request):
    try:
        # DEBUG: Show raw payload from Django
        raw_body = await request.json()
        #print("\n🛠 RAW REQUEST BODY:")
        #print(raw_body)

        # Prepare the input for the RAG chain using msg as the topic
        response = rag_chain.invoke({
            "input": "Can you create my personalized study guide?",
            "topic": input_data.msg,  # Topic from input
            "motivation_level": input_data.motivation_level,  # Motivation level from input
            "study_hours_per_week": input_data.study_hours,  # Study hours from input
            "preferred_learning_style": input_data.preferred_learning_style  # Learning style from input
        })

        # Debugging: Print the entire response
        logging.info(f"Response from RAG chain: {response}")

        # Check if 'answer' is in the response
        if isinstance(response, dict) and "answer" in response:
            return JSONResponse(content={"response": response["answer"]})
        else:
            return JSONResponse(content={"response": "No answer found.", "full_response": response})

    except Exception as e:
        logging.error(f"Error in /chatbot-advice: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    

 
