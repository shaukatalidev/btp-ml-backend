import asyncio
import json
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import nltk
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import datetime
import ffmpeg
import speech_recognition as sr
import cv2
import numpy as np
import imutils
import easyocr
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import tempfile
from googletrans import Translator
import subprocess
import imageio_ffmpeg
import nest_asyncio


# Initialize the NLTK stemmer
stemmer = nltk.PorterStemmer()


class StemmingRequest(BaseModel):
    text: str


app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_status():
    return {"status": "running"}


# @app.post("/stemmer", response_model=str)
# async def preprocess_queries(request: StemmingRequest):
#     text = request.text.lower().strip()
#     text = stemmer.stem(text)
#     return text


# Request model
class ProcessRequest(BaseModel):
    texts: List[str]


def preprocess_queries(texts):
    text = texts.lower().strip()
    text = stemmer.stem(text)
    return text


# # Step 4: Load ai4bharat's Indic-BERT model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModel.from_pretrained("ai4bharat/indic-bert")


# Custom embedding model class for BERTopic
class IndicBERTEmbedding:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed(self, documents):
        # Tokenize and encode the inputs
        inputs = self.tokenizer(
            documents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            # Get the embeddings from the model
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(
            dim=1
        )  # Average the token embeddings to get sentence embeddings
        return embeddings.numpy()  # Return as numpy array for compatibility


# Instantiate the embedding model
embedding_model = IndicBERTEmbedding(model, tokenizer)


@app.post("/process", response_model=List[str])
async def pos_queries(request: ProcessRequest):
    # Step 1: Load the dataset from request
    rawdatas = request.texts
    datas = [preprocess_queries(rawdata) for rawdata in rawdatas]

    # Step 2: Convert to DataFrame for easier handling
    df = pd.DataFrame({"question": datas})

    # Step 3: Generate embeddings using Indic-BERT
    questions = df["question"].tolist()
    embeddings = embedding_model.embed(questions)

    # Step 4: Apply KMeans clustering on the embeddings
    num_clusters = 5  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)

    # Step 5: Identify the most representative question for each cluster
    clusters_multilingual = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters_multilingual:
            clusters_multilingual[label] = questions[i]

    # Step 6: Return the representative questions
    representative_questions_list = [
        question for _, question in clusters_multilingual.items()
    ]

    return representative_questions_list


def rdm():
    return str(datetime.datetime.now().timestamp())


def convertToWAV(file: UploadFile):
    os.makedirs("uploads", exist_ok=True)  # ensure directory exists

    filename = f"{datetime.datetime.now().timestamp()}"
    ext = re.split("[/;]", file.content_type)[1]
    input_path = f"uploads/{filename}.{ext}"
    output_path = f"uploads/{filename}.wav"

    with open(input_path, "wb") as f:
        f.write(file.file.read())

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error:", e.stderr.decode())
        raise

    os.remove(input_path)  # cleanup original file
    return output_path


# Optional: keep if you need for torchaudio models
def preprocess_audio(file_path, processor, sample_rate=16000):
    speech_array, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        speech_array = resampler(speech_array)
    input_values = processor(
        speech_array.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate
    ).input_values
    return input_values


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...), source: str = Form("en"), language: str = Form("en")
):
    print(f"Requested translation: {source} → {language}")

    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an audio file."
        )

    print(f"Uploaded file: {file.filename} | Type: {file.content_type}")

    try:
        # Convert MP3 (or other) to WAV
        if file.content_type != "audio/wav":
            file_path = convertToWAV(file)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(await file.read())
                file_path = temp_audio.name

        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source_audio:
            audio_data = recognizer.record(source_audio)
        transcript = recognizer.recognize_google(audio_data)
        print("Transcript:", transcript)

        # Translate
        # translator=Translator()
        transcribeconv = await Translator().translate(transcript, src="en", dest=source)
        translated_text = await Translator().translate(
            transcribeconv.text, src=source, dest=language
        )

        print("Translated Text:", translated_text.text)

        return {
            "transcript": translated_text.text,
            "Transcriptconv": transcribeconv.text,
            "translated": transcript,
        }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(
            status_code=500, detail="Transcription or translation failed."
        )


@app.post("/mcq-analysis")
async def mcq_analysis(
    file: UploadFile = File(...),
    rollNumbers: str = Form(...),
    class_limit: Optional[int] = Form(...),
):
    initial_expected_roll_numbers = json.loads(
        rollNumbers
    )  # Convert JSON string back to a list

    print("initial_expected_roll_numbers:", initial_expected_roll_numbers)
    reader = easyocr.Reader(["en"])
    current_expected_roll_numbers = initial_expected_roll_numbers
    extracted_roll_numbers = set()
    extracted_mcq_with_roll = set()

    # Read the image from the uploaded file
    image_data = await file.read()
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    while current_expected_roll_numbers:
        read_image = image_np
        image_path, gray, blackhat, rectKern = load_and_preprocess_image(read_image)
        gradX = apply_sobel_and_normalize(blackhat)
        thresh = apply_morphological_transformations(gradX, rectKern)
        imgFiltered = filter_components(thresh)
        lpCnt, roi = find_and_extract_roi(imgFiltered, blackhat)
        lpTxt = read_text_from_roi(roi, reader)
        draw_bounding_boxes_and_text(image_path, lpCnt, lpTxt)
        (
            current_expected_roll_numbers,
            temp_extracted_roll_number,
            temp_extracted_mcq_with_roll,
        ) = extract_text_and_display(
            image_path, current_expected_roll_numbers, reader, class_limit
        )

        current_expected_roll_numbers = set(current_expected_roll_numbers)

        for roll in temp_extracted_roll_number:
            extracted_roll_numbers.add(roll)

        print("temp_extracted_mcq", temp_extracted_mcq_with_roll)
        for roll_mcq in temp_extracted_mcq_with_roll:
            extracted_mcq_with_roll.add(roll_mcq)

        print("extracted_mcq_with_roll", extracted_mcq_with_roll)
        if not current_expected_roll_numbers:
            return JSONResponse(
                content={
                    "message": "All roll numbers have been detected.",
                    "extracted_roll_numbers": list(extracted_roll_numbers),
                    "extracted_mcq_with_roll": list(extracted_mcq_with_roll),
                }
            )
        else:
            return JSONResponse(
                content={
                    "message": f"Please upload an image with the following missing roll numbers: {current_expected_roll_numbers}",
                    "extracted_roll_numbers": list(extracted_roll_numbers),
                    "extracted_mcq_with_roll": list(extracted_mcq_with_roll),
                }
            )


def load_and_preprocess_image(image_path):
    # img = cv2.imread('question1v5.jpg')
    gray = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 150))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    return image_path, gray, blackhat, rectKern


def apply_sobel_and_normalize(blackhat):
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = 255 * (
        (np.absolute(gradX) - np.min(gradX)) / (np.max(gradX) - np.min(gradX))
    )
    return gradX.astype("uint8")


def apply_morphological_transformations(gradX, rectKern):
    gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.adaptiveThreshold(
        gradX, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.erode(thresh, None, iterations=3)
    thresh = cv2.dilate(thresh, None, iterations=3)
    return thresh


def filter_components(thresh, min_size=3000, max_size=20000):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh)
    sizes = stats[1:, -1]
    imgFiltered = np.zeros_like(thresh)
    for i in range(0, nb_components - 1):
        if min_size <= sizes[i] <= max_size:
            imgFiltered[output == i + 1] = 255
    return imgFiltered


def find_and_extract_roi(imgFiltered, blackhat):
    cnts = cv2.findContours(imgFiltered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    lpCnt, roi = [], []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if 1 <= ar <= 2:
            lpCnt.append(c)
            roi.append(blackhat[y : y + h, x : x + w])
    return lpCnt, roi


def read_text_from_roi(roi, reader):
    return [reader.readtext(region)[0][-2] for region in roi if reader.readtext(region)]


def draw_bounding_boxes_and_text(img, lpCnt, lpTxt):
    for i in range(len(lpTxt)):
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt[i]))
        box = box.astype("int")
        cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
        (x, y, w, h) = cv2.boundingRect(lpCnt[i])
        cv2.putText(
            img, lpTxt[i], (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
        )
    cv2.imwrite("ullaspullamydata.jpg", img)


def extract_text_and_display(img, expected_roll_numbers, reader, class_limit):
    result = reader.readtext(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    extracted_roll_numbers = set()
    extracted_mcq_with_roll = set()

    for i, (bbox, text, prob) in enumerate(result):
        (top_left, bottom_right) = (tuple(map(int, bbox[0])), tuple(map(int, bbox[2])))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Handle the detection of both roll number and MCQ answer
        if len(text) == 1 and text in "ABCD":
            prev_text = result[i - 1][1]
            if prev_text.isdigit():
                roll_number = int(prev_text)
                if 1 <= roll_number <= class_limit:
                    text = f"{roll_number} {text}"

        if not text.isdigit() and not text.isalpha():
            roll_number, mcq_answer = (
                "".join(filter(str.isdigit, text)),
                "".join(filter(str.isalpha, text)),
            )

            # If the roll number is detected, clean it by removing leading zeroes for single digits
            if roll_number:
                roll_number = str(int(roll_number))  # This removes leading zeros
                roll_number = int(
                    roll_number
                )  # Convert to integer for boundary checking
                if 1 <= roll_number <= class_limit:
                    extracted_roll_numbers.add(roll_number)

            if roll_number and mcq_answer:
                extracted_mcq_with_roll.add((roll_number, mcq_answer))

    # Find missing roll numbers
    missing_roll_numbers = sorted(
        list(set(expected_roll_numbers) - extracted_roll_numbers), key=int
    )

    return (
        missing_roll_numbers,
        sorted(extracted_roll_numbers),
        sorted(extracted_mcq_with_roll),
    )
