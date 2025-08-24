from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
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
from openai import OpenAI as OpenAIClient
from helper import download_hugging_face_embeddings 
from prompt import prompt
from fastapi import Request


# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
my_openai_api_key = os.getenv('MY_OPENAI_API_KEY')


if not my_openai_api_key:
    raise ValueError("my_openai_api_key not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize FastAPI and logging
app = FastAPI(
    title="GoalTweaks : The Student Performance App you need",
    description="App for predicting scores, providing chatbot advice, and generating study plans",
    version="1.0.0"
)
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
openai_client = OpenAIClient(api_key=my_openai_api_key)

# Define request models
class PredictionInput(BaseModel):
    hours_studied: float
    previous_scores: float
    extracurricular: str  # "Yes" or "No"
    sleep_hours: float
    question_papers: int
    feedback: str = "I'm stressed and need better planning."  # optional

class ChatInput(BaseModel):
    subject: str
    user_id: str  
    predicted_score: float
    subject_weekly_study_hours: float
    motivation_level: str
    preferred_learning_style: str


class QuestionnaireInput(BaseModel):
    subjects: str = Field(..., min_length=1, description="Comma-separated list of subjects (e.g., 'Math,Science')")
    learning_style: Literal["visual", "auditory", "kinesthetic", "reading_writing"] = Field(
        ..., description="Preferred learning style"
    )
    goal: str = Field(..., min_length=1, description="Academic goal (e.g., 'Score 90% in finals')")

    # This is the total hours to study per week.
    hours_per_week: int = Field(..., ge=1, description="Hours available for study per week")

    hours_studied: int = Field(..., ge=0, description="Hours studied per week")
    sleep_hours: int = Field(..., ge=0, le=24, description="Average sleep hours per night")

    #This should be used to advice on whether to reduce the hours or not. 
    hours_for_extracurricular: int = Field(..., ge=0, description="Hours spent on extracurricular activities")

    # This should ask weekly papers solved.
    question_papers_solved: int = Field(..., ge=0, description="Number of question papers solved")

    study_habits: str = Field(..., min_length=1, description="Description of study habits")
    #Should be GPA
    previous_grades: int = Field(..., ge=0, le=100, description="Previous grades (0-100)")

    motivation_level: Literal["low", "medium", "high"] = Field(
        ..., description="Motivation level"
    )

class StudyPlanResponse(BaseModel):
    study_plan: str
     

# Load models
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="students-score-padi", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = OpenAI(temperature=0.4, max_tokens=1500)


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
    return {"message": "Welcome to GoalTweaks : Set your academic goals, tweak habits and achieve them"}


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
async def get_chatbot_advice(input_data: ChatInput):
    try:
        # Create the input prompt for the OpenAI model
        input_prompt = (
            f"Create a personalized study guide for the subject: {input_data.subject} "
            f"with a predicted score of {input_data.predicted_score}%. "
            f"Consider the motivation level: {input_data.motivation_level} and "
            f"preferred learning style: {input_data.preferred_learning_style}."
            "Please format the output as a single <div> element containing the study guide, "
            "including appropriate HTML tags for headings(starting from <h4>), lists, and tables. "
            "Do not include a full HTML document structure."
        )

        # Call OpenAI LLM
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a study assistant creating personalized study guides."},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        # Access the response correctly
        study_guide = response.choices[0].message.content.strip()

        if not study_guide:
            logging.warning("No study guide content returned.")
            return JSONResponse(content={"study_guide": "No content available."}, status_code=200)

        logging.info("Study Guide generated successfully.")
        logging.info(study_guide)
        return JSONResponse(content={"study_guide": study_guide})

                     
    except Exception as e:
        logging.error(f"Error in /chatbot-advice: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")





# Study plan endpoint
@app.post("/study-plan", response_model=StudyPlanResponse)
async def create_study_plan(data: QuestionnaireInput):
    try:
        # Print incoming data for debugging
        print("Received data:", data)

        # Split subjects into a list
        subjects = [s.strip() for s in data.subjects.split(",")]
        print("Parsed subjects:", subjects)

        num_subjects = len(subjects)
        if num_subjects == 0:
            print("No subjects provided.")
            raise HTTPException(status_code=400, detail="No subjects provided")

        # Calculate hours per subject
        available_hours = max(1, data.hours_per_week - data.hours_for_extracurricular)
        logging.info(f"Available hours: {available_hours}")

        hours_per_subject = available_hours // num_subjects
        extra_hours = available_hours % num_subjects
        # print("Hours per subject:", hours_per_subject, "Extra hours:", extra_hours)
        logging.info(f"Hours per subject: {hours_per_subject}, Extra hours: {extra_hours}")

        # Define study days
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        max_daily_hours = 24 - data.sleep_hours - (data.hours_for_extracurricular // 6)
        max_daily_hours = max(2, min(max_daily_hours, 6))
        # print("Max daily hours:", max_daily_hours)
        logging.info(f"Max daily hours: {max_daily_hours}")

        # Create prompt for OpenAI LLM
        study_plan_prompt = (
            f"Create a detailed weekly study plan for a student with the following details:\n"
            f"- Subjects: {', '.join(subjects)}\n"
            f"- Academic Goal: {data.goal}\n"
            f"- Learning Style: {data.learning_style}\n"
            f"- Available Study Hours per Week: {data.hours_per_week}\n"
            f"- Current Study Hours: {data.hours_studied}\n"
            f"- Sleep Hours per Night: {data.sleep_hours}\n"
            f"- Extracurricular Hours per Week: {data.extracurricular}\n"
            f"- Question Papers Solved: {data.question_papers_solved}\n"
            f"- Study Habits: {data.study_habits}\n"
            f"- Previous Grades: {data.previous_grades}%\n"
            f"- Motivation Level: {data.motivation_level}\n"
            f"Requirements:\n"
            f"- Allocate approximately {hours_per_subject} hours per subject, with {extra_hours} extra hours distributed to some subjects.\n"
            f"- Spread study sessions across {len(days)} days ({', '.join(days)}), with no more than {max_daily_hours} study hours per day.\n"
            f"- Tailor study activities to the student's {data.learning_style} learning style (e.g., videos for visual, podcasts for auditory, hands-on for kinesthetic, notes for reading/writing).\n"
            f"- Include specific recommendations to improve based on {data.study_habits} and {data.motivation_level} motivation.\n"
            f"- Use a friendly, engaging tone, incorporating Ghanaian Pidgin phrases (e.g., 'Chale, make you try dis!') for motivation.\n"
            f"- Output a clear schedule with subjects, hours, days, and activities, followed by general study tips.\n"
            f"- Ensure the plan is practical and aligns with the student's goal of '{data.goal}'.\n"
        )

        # Print the generated prompt for debugging
        # print("Generated study plan prompt:", study_plan_prompt)

        # Log the generated prompt
        logging.info(f"Generated study plan prompt: {study_plan_prompt}")

        # Call OpenAI LLM
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a study assistant creating personalized study plans."},
                    {"role": "user", "content": study_plan_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            study_plan = response.choices[0].message.content.strip()
            print("Generated study plan:", study_plan)
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating study plan: {str(e)}")

        return StudyPlanResponse(study_plan=study_plan )

    except Exception as e:
        logging.error(f"Error in /study-plan: {str(e)}")
        # print("Error details:", str(e))  # Print error details for debugging
        raise HTTPException(status_code=500, detail=f"Error generating study plan: {str(e)}")