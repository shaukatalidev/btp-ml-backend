import json
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import datetime
import ffmpeg
import speech_recognition as sr
import cv2
import numpy as np
import imutils
import easyocr
from openpyxl import Workbook

# Initialize the NLTK stemmer
stemmer = nltk.PorterStemmer()


class ProcessRequest(BaseModel):
    texts: List[str]


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
def preprocess_queries(texts):
    text = texts.lower().strip()
    text = stemmer.stem(text)
    return text


@app.post("/process", response_model=List[str])
async def pos_queries(request: ProcessRequest):
    rawdatas = request.texts
    datas = []
    for rawdata in rawdatas:
        datas.append(preprocess_queries(rawdata))
    representative_questions = []
    df = pd.DataFrame({"question_text": [], "cluster": []})
    # print(request.texts)
    df["question_text"] = datas
    tfidfVectorizer = TfidfVectorizer()
    query_embeddings = tfidfVectorizer.fit_transform(df["question_text"])
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(query_embeddings)
    df["cluster"] = kmeans.labels_
    for cluster_id in range(num_clusters):
        cluster_data = df.loc[df["cluster"] == cluster_id]
        centroid = kmeans.cluster_centers_[cluster_id]
        similarities = cosine_similarity(
            tfidfVectorizer.transform(cluster_data["question_text"]), [centroid]
        )
        representative_question = cluster_data.iloc[similarities.argmax()][
            "question_text"
        ]
        representative_questions.append(representative_question)
    return representative_questions


def rdm():
    return str(datetime.datetime.now().timestamp())


def convertToWAV(file):
    filename = f"{datetime.datetime.now().timestamp()}"
    inloc = f"uploads/{filename}.{re.split('[/;]',file.content_type)[1]}"
    outloc = f"uploads/{filename}.wav"
    with open(inloc, "wb") as f:
        f.write(file.file.read())
    ffmpeg.input(inloc).output(outloc).run()
    return outloc


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not re.match("audio/", file.content_type):
        raise HTTPException(status_code=400, detail="File type not supported")

    if not re.match("audio/wav", file.content_type):
        file_path = convertToWAV(file)
    else:
        file_path = file.filename

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        return {"transcript": transcript}
    except Exception as e:
        print("error: ", str(e))
        raise HTTPException(status_code=500, detail="Unable to transcribe")


@app.post("/mcq-analysis")
async def mcq_analysis(file: UploadFile = File(...), rollNumbers: str = Form(...)):
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
        ) = extract_text_and_display(image_path, current_expected_roll_numbers, reader)

        current_expected_roll_numbers = set(current_expected_roll_numbers)

        for roll in temp_extracted_roll_number:
            extracted_roll_numbers.add(roll)

        for roll_mcq in temp_extracted_mcq_with_roll:
            extracted_mcq_with_roll.add(roll_mcq)

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


def extract_text_and_display(img, expected_roll_numbers, reader):
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

        if len(text) == 1 and text in "ABCD":
            prev_text = result[i - 1][1]
            if prev_text.isdigit() and 0 <= int(prev_text) <= 40:
                text = f"{prev_text} {text}"
        if not text.isdigit() and not text.isalpha():
            roll_number, mcq_answer = (
                "".join(filter(str.isdigit, text)),
                "".join(filter(str.isalpha, text)),
            )
            if roll_number and mcq_answer:
                extracted_roll_numbers.add((roll_number))
                extracted_mcq_with_roll.add((roll_number, mcq_answer))
                # print(f"Roll Number: {roll_number}, MCQ Answer: {mcq_answer}")
    # print('extracted_roll_no:', extracted_roll_numbers)
    missing_roll_numbers = sorted(
        list(set(expected_roll_numbers) - extracted_roll_numbers), key=int
    )

    # print("Expected Roll Numbers:", expected_roll_numbers)
    # print("Extracted Roll Numbers:", extracted_roll_numbers)
    # print("Missing Roll Numbers:", missing_roll_numbers)

    return missing_roll_numbers, extracted_roll_numbers, extracted_mcq_with_roll
