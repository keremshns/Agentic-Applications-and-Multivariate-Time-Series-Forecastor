import os
import io
from io import BytesIO, StringIO 
import base64
from pathlib import Path
import json
import cv2
import whisper
from PIL import Image
from typing import TextIO

from PIL import Image
import base64
import urllib.request
import urllib.parse

from pytubefix import YouTube
import glob
from moviepy.editor import VideoFileClip
from _collections_abc import Iterator
import textwrap
import webvtt

import time
from pydantic import BaseModel, Field
from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.enum import EnumOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

from operator import itemgetter
from langchain_openai import ChatOpenAI
from openai import OpenAI


os.environ["OPENAI_API_KEY"] = ""

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ['USER_AGENT'] = 'myagent'

#CONSTANTS
model_finetuned = ""
model_finetuned_pavo = ""
model_finetuned_suerox = ""  # 
model_instruct = "gpt-3.5-turbo-instruct"
model_chat = "gpt-3.5-turbo"
model_4o_mini = "gpt-4o-mini"
model_4o = "gpt-4o"
model_token_limit = 4096

#FUNCTIONS
def download_video(url_link, path_to_download):
    urllib.request.urlretrieve(url_link, path_to_download) 

def download_video_youtube(url):
    video_meta = {}
    
    if 1:
        yt = YouTube(url)
        video_meta["title"] = yt.title
        video_meta["author"] = yt.author
        video_meta["description"] = yt.description
        video_meta["captions"] = yt.captions

        ys = yt.streams.get_highest_resolution()
        ys.download()
    
    return video_meta

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def get_video_path(video_url, path):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = os.path.join(path, urllib.parse.urlparse(video_url).path)
    print(filepath)
    if len(filepath) > 0:
        return filepath #CONVERT TO ALL PLATFORM STRING
    else:
        print("AN ERROR OCCURED WHILE GETTING VIDEO PATH.") 

# function `extract_and_save_frames_and_metadata``:
#   receives as input a video and its transcript
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata(
        path_to_video, 
        vtt_transcribed, 
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    # load transcript using webvtt
    if len(vtt_transcribed) != 0:
        try:
            trans = webvtt.read(vtt_transcribed)
        except:
            print("There was a problem reading vtt.")
    else:
        print("vtt path is empty because there were no audio files.")
        trans = 0
    
    delete_files_in_directory(path_to_save_extracted_frames) #BEFORE WRITING FRAME MAKE SURE THE FILE IS
    
    if trans:
        # iterate transcript file
        # for each video segment specified in the transcript file
        for idx, transcript in enumerate(trans):
            # get the start time and end time in seconds
            start_time_ms = str2time(transcript.start)
            end_time_ms = str2time(transcript.end)
            # get the time in ms exactly 
            # in the middle of start time and end time
            mid_time_ms = (end_time_ms + start_time_ms) / 2
            # get the transcript, remove the next-line symbol
            text = transcript.text.replace("\n", ' ')
            # get frame at the middle time
            video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
            success, frame = video.read()
            if success:
                # if the frame is extracted successfully, resize it
                image = maintain_aspect_ratio_resize(frame, height=350)
                # save frame as JPEG file
                img_fname = f'frame_{idx}.jpg'
                img_fpath = os.path.join(
                    path_to_save_extracted_frames, img_fname
                )

                cv2.imwrite(img_fpath, image)

                # prepare the metadata
                metadata = {
                    'extracted_frame_path': img_fpath,
                    'transcript': text,
                    'video_segment_id': idx,
                    'video_path': path_to_video,
                    'mid_time_ms': mid_time_ms,
                }
                metadatas.append(metadata)
            else:
                print(f"ERROR! Cannot extract frame: idx = {idx}")

    else:   
        freq = 10 #NUM FRAMES TO EXTRACT
        fps = video.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = frame_count/fps
        period = duration_ms / freq
            
        idx = 0
        curr_time = 0
        # iterate transcript file
        # for each video segment specified in the transcript file
        while(duration_ms > curr_time):
                
            # get frame at the middle time
            video.set(cv2.CAP_PROP_POS_MSEC, curr_time)
            success, frame = video.read()
            if success:
                # if the frame is extracted successfully, resize it
                image = maintain_aspect_ratio_resize(frame, height=350)
                # save frame as JPEG file
                img_fname = f'frame_{idx}.jpg'
                img_fpath = os.path.join(
                        path_to_save_extracted_frames, img_fname
                    )

                cv2.imwrite(img_fpath, image)

                    # prepare the metadata
                metadata = {
                        'extracted_frame_path': img_fpath,
                        'transcript': "",  #NO TEXT SO IT IS EMPTY 
                        'video_segment_id': idx,
                        'video_path': path_to_video,
                        'mid_time_ms': "EACH TIME FRAME",
                    }
                metadatas.append(metadata)

                idx = idx+1
                curr_time = curr_time + period
            else:
                print(f"ERROR! Cannot extract frame: idx = {idx}")
                idx = idx+1
                curr_time = curr_time + period
               

    # save metadata of all extracted frames
    fn = os.path.join(path_to_save_metadatas, 'metadatas.json')
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

# a help function that helps to convert a specific time written as a string in format `webvtt` into a time in miliseconds
def str2time(strtime):
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]
    # get the corresponding time as total seconds 
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds
    
def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

# helper function to convert transcripts generated by whisper to .vtt file
def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}

def prompt_func(data_dict):
    
    #DEFINE MODEL OUTPUT FORMATS.
    class Output_Format(BaseModel):
        Condition: str = Field(description="""Assigned condition as string. Possible values are "Approved" or "Rejected". """)
        Reason: str = Field(description="""Reason for the assigned condition. Also gives reference from the actual sentence in the context.""" )

    messages = []
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    
    #DEFINE PARSERS
    json_parser =JsonOutputParser(pydantic_object=Output_Format)
    format_instructions=json_parser.get_format_instructions()
    
    #RAG SOURCE
    docs = []
    for file in os.scandir(r"C:\Users\kerem\Desktop\\Terms Documents\Text Files for RAG"):
            
        with open(file, encoding="utf-8") as f:
            read_text = f.read()

        #recursive text splitter (this might be better to preserve info)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        texts = text_splitter.split_text(read_text)

        num_texts = len(texts)
        for t in texts[:num_texts]:
            docs.append(Document(page_content=t))

        #print(docs)

    #Retrieve and generate using the relevant snippets of the blog.
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Adding the text message for analysis
    #- A detailed description of the visual elements in the image. THIS CAN BE ADDED TO PROMPT

    text_message = {
        "type": "text",
        "text": (f"""SYSTEM INSTRUCTIONS:\n
        You are trained to analyze and detect if the given text and image input are compatible with the terms, policies and design guidelines which are provided in the context.\n
        Your task is to analyze and interpret the text and image. Determine if they are compatible with the terms, policies and design guidelines which are provided in the context.\n 
        Please use your extensive knowledge and analytical skills to provide a decision which will be "Approved" or "Rejected" according to the all information in retrieved context.\n
        If the input text is empty, just check the provided images to determine if they should be "Approved or Rejected". If there are no images provided to you just use the text input.
        if it should be "Approved or Rejected".\n
                 
        
        Your job has 4 steps:
            1- Analyze the given text and image inputs.
            2- Check if they are compatible with the provided context information. In other words, detect if text or image input breaches or violates any term, policy or guideline that is provided in the context. 
            3- Determine if it should be "Approved" or "Rejected" according to retrieved context. Return a "Condition" in a single string as either "Approved" or "Rejected".
            4- Provide the reason for the "Condition" that is assigned. Also you should indicate your reference from the actual context.

        Input Text: {formatted_texts}
        Context: {retriever}
        Format Instructions: {format_instructions}
        """
        ),
    }
    messages.append(text_message)

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}" #ONLY GIVING ONE IMAGE FOR NOW
            },
        }
        messages.append(image_message)

    return [HumanMessage(content=messages)]

def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    #image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    #display(HTML(image_html))
    #print(img_base64)
    decoded_img = base64.b64decode(img_base64)
    image = Image.open(BytesIO(decoded_img))
    image.show()

#----------------------------------------------------
def build_context_list(metadatas):
        images = []
        texts = []
        for data in metadatas:
            text = data["transcript"]
            texts.append(text)

            image_path = data["extracted_frame_path"]
            with open(image_path, "rb") as image:
                
                image = image.read()
                if is_base64(image):
                        # Resize image to avoid OAI server error
                    images.append(
                        resize_base64_image(image, size=(250, 250))
                        )  # base64 encoded str
                else:
                    image = base64.b64encode(image)
                    images.append(
                        resize_base64_image(image, size=(250, 250))
                        )  # base64 encoded str
        
        return {"images": list(images), "texts": list(texts)}
    
        #EXPRESSIONS FOR TESTING IMAGES AND TEXT LIST BUILDING
        #print(len(images))
        #print(len(texts))
        #print("\n", images)
        #print("\n", texts, "\n")

#DEFINE SENTIMENT MODEL
def SentimentModel(rag_source_path, input_text_list):
    #TIME CALCULATIONS FOR THE MODEL
    start = time.time()
    
    #MODELS
    fine_llm = ChatOpenAI(model=model_finetuned_suerox)
    llm = ChatOpenAI(model=model_4o_mini)   #OpenAI(gpt-3.5-turbo instruct) or ChatOpenAI (gpt-3.5-turbo)

    #TEXT SPLITTERS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    #rec_text_split_test = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)

    #DEFINE MODEL OUTPUT FORMATS.
    class Output_Format(BaseModel):
        Sentiment: str = Field(description="Assigned sentiment as string. Possible values are 1 or -1 or 0")
    class Sentiments(Enum):
        Positive = "1"
        Negative = "-1"
        Neutral = "0"

    #DEFINE PARSERS
    json_parser =JsonOutputParser(pydantic_object=Output_Format)
    enum_parser = EnumOutputParser(enum=Sentiments)

    
    #DEFINE PROMPTS AND TEMPLATES
    #TEMPLATES
    rag_template = """Your job is to analyze and detect the sentiment of given text input. Use the following pieces of retrieved context to answer the question.\n
    Analyze the following product review in the user prompt and determine if the sentiment is: Positive, Negative or Neutral. Return answer in a single string as either 1, -1 or 0\n
    
    Question: {text_input} 

    Context: {context}

    Instructions: {instructions}
    """

    fine_template = """You are trained to analyze and detect the sentiment of given text.Analyze the following product review in the user prompt and determine if\n
                        the sentiment is: Positive, Negative or Neutral.Return answer in single word as either Positive, Negative or Neutral.\n
                        INPUT: {text}
                        OUTPUT FORMAT: {format_instructions}"""

    #PROMPTS
    rag_prompt = PromptTemplate(template=rag_template,
                                    input_variables=["text_input", "context"]
                                    ).partial(instructions=enum_parser.get_format_instructions())  #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.

    fine_prompt = PromptTemplate(template=fine_template,
                                    input_variables=["text"],
                                    partial_variables={"format_instructions": json_parser.get_format_instructions()})  #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.

    #BUILD CHAINS
    rag_chain = (
        rag_prompt
        | llm
        | enum_parser
        )

    fine_chain = (
        {
            "text": itemgetter("docs")
        }
        |fine_prompt
        | fine_llm
        | json_parser
        )
    
    

    #TAKE RAG SOURCE AS JSON AND CONVERT TO STRING FORMAT FOR BETTER PERFORMANCE
    data_json = []
    with open(rag_source_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data_json.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line}")
                    continue
            
    #print(data_json[0][0]["messages"])  RAG DATA SHOULD BE ACCORDING TO OUR OPENAI TRAINING FORMAT FOR NOW// CAN CHANGE LATER !!!

    # CONVERT JSON TO TEXT FOR RAG PERFORMANCE
    text_l = []
    for elem in data_json[0]:
        content = elem["messages"][1]["content"]
        sent = elem["messages"][2]["content"]

        # Prepare text for embedding
        text_to_embed = f""" Content: <{content}> is the given Text Input. Sentiment: <{sent}> is the sentiment of the given text input."""
        text_l.append(text_to_embed)


    texts = "\n".join(text_l)
    texts = text_splitter.split_text(texts)

    num_texts = len(texts)
    docs = [Document(page_content=t) for t in texts[:num_texts]]

    #FORMING VECTOR DATABASE
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()  

    model_score = []

    for text in input_text_list:   #for index, row in df.iterrows(): LOOP 
        
        #print(text) to check text if its appropriate
        texts = text_splitter.split_text(text)
        num_texts = len(texts)
        docs = [Document(page_content=t) for t in texts[:num_texts]]
        results = vectorstore.similarity_search_with_relevance_scores(text, k=1)
        context = "\n\n".join([doc.page_content for doc, _score in results])
        #print(docs)

        if len(results) == 0 or results[0][1] < 0.8:
            #RUN CHAINS
            data = fine_chain.invoke({"docs": docs})
            
            #comparison_output = json.dumps(data, indent=2)
            #print(type(data))
            #print(data)
            
            
            try:
                # Check if data is a valid JSON string
                if isinstance(data, dict):
                    model_score.append(data.get("Sentiment"))
                elif type(data) == int or type(data) == str:
                    print("UNEXPECTED OUTPUT FINETUNE OUTPUT IS NOT DICT")
                    model_score.append(data)
                else:
                    model_score.append("NaN")
            except:
                model_score.append("NaN")
                print("error parsing output")

        elif results[0][1] > 0.8:

            #RUN CHAINS
            data = rag_chain.invoke({"context": context , "text_input": docs})
            
            #comparison_output = json.dumps(data, indent=2)
            #print(type(data))
            #print(data)
            
            
            try:
                # Check if data is a valid JSON string
                if isinstance(data.value, str):
                    model_score.append(data.value)
                elif type(data.value) == int:
                    print("UNEXPECTED OUTPUT FINETUNE OUTPUT IS NOT DICT")
                    model_score.append(str(data.value))
                else:
                    model_score.append("NaN")
            except:
                model_score.append("NaN")
                print("error parsing output")

    end = time.time()
    print(end - start)

    return model_score[0]

#VIDEO APPROVAL FUNCTION
#------------------------------------------------------

def Video_Approval(vid_filepath, vid_dir):  #MAIN FUNCTION

    delete_files_in_directory(r"C:\Users\kerem\Desktop\\Approval_Model\videos")
    
    #cloud_filepath = urllib.parse.urlparse(vid_url).path
    #filename = os.path.basename(cloud_filepath)
    #new_filepath = os.path.join(vid_dir, filename)
    #download_video(vid_url, new_filepath)
    #print(new_filepath)
    
    #DEFINE FINAL VIDEO FILEPATH
    vid_file = vid_filepath
    new_filepath = vid_filepath
   
    #OPTION: RUN TRANSCRIBE ON AUDIO
    #-------------------------------
    # declare where to save .mp3 audio

    path_audio_file = os.path.join(vid_dir, 'audio.mp3')
    path_audio_file= r'{}'.format(path_audio_file) #convert to raw string
    print(path_audio_file)

     #DELETE EXISTING AUDIO FILE IF IT EXISTS
    if os.path.exists(path_audio_file):
        os.remove(path_audio_file)

    # extract mp3 audio file from mp4 video video file
    clip = VideoFileClip(new_filepath)    
        
    try:
        clip.audio.write_audiofile(path_audio_file)
        audio_written=1
    except:
        print("there was an error writing the audio")
        audio_written=0   

    if audio_written==1:
        print("Using Whisper...")
        #DO TRANSCRIPTIONS 
        #whisp_model = whisper.load_model("turbo")
    
        client = OpenAI(organization="", project="")

        audio_file= open(path_audio_file, "rb")
        translation = client.audio.transcriptions.create(
                                                        model="whisper-1", 
                                                         file=audio_file,
                                                         response_format="vtt",
                                                         temperature=0
                                                        )
        #print(translation)
        
        vid_vtt = translation
        # path to save generated transcript of video
        path_to_vid_vtt = os.path.join(vid_dir, 'generated_video.vtt')
        # write transcription to file
        with open(path_to_vid_vtt, 'w', encoding='utf-8') as f:
            f.write(vid_vtt)
    else:
        path_to_vid_vtt = "" #NO AUDIO SO CANNOT PORDUCE VTT
        print("no audio")

    #RUN LVLM INFERENCE TO INTERPRET VIDEO FRAMES
    #---------------------------------------------------------------------------------
    # output paths to save extracted frames and their metadata 
    extracted_frames_path = os.path.join(vid_dir, 'extracted_frame')
    metadatas_path = vid_dir

    # create these output folders if not existing
    Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
    Path(metadatas_path).mkdir(parents=True, exist_ok=True)

    # call the function to extract frames and metadatas
    metadatas = extract_and_save_frames_and_metadata(
                    vid_file, 
                    path_to_vid_vtt,
                    extracted_frames_path,
                    metadatas_path,
                )

    #load metadata files
    vid_metadata_path = r'C:\Users\kerem\Desktop\\Approval_Model\Video_Test_Area\metadatas.json'
    # Open the JSON file in read mode
    with open(vid_metadata_path, 'r') as file:
        vid_metadata  = json.load(file)
    #print(vid_metadata)

    # collect transcripts and image paths
    vid_trans = [vid['transcript'] for vid in vid_metadata]
    vid_img_path = [vid['extracted_frame_path'] for vid in vid_metadata]

    #MAY BE USED TO IMPROVE PERFORMANCE, WONT BE USED FOR NOW
    #---------------------------------------------------------------------------------------
    #WE SELECT AN N, and make transcripts such that N/2 before a specific transcrip and N/2 after a specific transcript are added to curent one to provide more context. 
    #(NORMALLY DONE WHERE TRANSCRIPT IS AVAILABLE BUT CAN BE IMPLEMENTED FOR BETTER PREFORMANCE WHERE WE USE WHISPER)
    """Notes:
    - We observe that the transcripts of frames extracted from video1 are usually fragmented and even an incomplete sentence.
    E.g., four more was just icing on the cake for a. Thus, such transcripts are not meaningful and are not helpful for retrieval.
    In addition, a long transcript that includes many information is also not helpful in retrieval. A naive solution to this issue is to augment such a transcript with the transcripts of n neighboring frames. 
    It is advised that we should pick an individual n for each video such that the updated transcripts say one or two meaningful facts.
    - It is ok to have updated transcripts of neighboring frames overlapped with each other.
    - Changing the transcriptions which will be ingested into vector store along with their corresponding frames will affect directly the performance. It is advised that one needs to do diligent to experiment with one's data to get the best performance."""

    # for video1, we pick n = 3
    #n = 3
    #updated_vid_trans = [
    #' '.join(vid_trans[i-int(n/2) : i+int(n/2)]) if i-int(n/2) >= 0 else
    #' '.join(vid_trans[0 : i + int(n/2)]) for i in range(len(vid_trans))
    #]

    #NEW METADATA IF CONCATENATING TRANSCRIPTS CAN UPDATE PERFORMANCE
    #vid_updated_transcripts = []
    # also need to update the updated transcripts in metadata
    #for i in range(len(updated_vid_trans)):
    #    vid_updated_transcripts.append({'transcript': updated_vid_trans[i]})

    #print(vid_metadata_updated_transcripts)
#----------------------------------------------------------------------------------------
    
    #***NOW WE HAVE FRAMES AND RELATED TRANSCRIPT WE CAN START BUILDING THE MODEL AND CHAINS***
#__________________________________________________________________________________________________
    
    #CHECK FOR PRETRAINED CLIP MODELS IF NEEDED
    #print(open_clip.list_pretrained()) #TRAINED CLIP MODELS

    #SAMPLE MODEL USED IN EXAMPLE; WE CAN TRY IT
    #model_name = "ViT-g-14" #LARGE MODEL W HUGH PERF BUT PROBABLY SLOW
    #checkpoint = "laion2b_s34b_b88k"

    # Embedding Pretrained
    #clip_embd = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

    rejected_keywords = [
        " fuck",
        " shit",
        " cunt",
        " asshole",
        " bitch",
        " fucker",
        " wanker",
        " nigger",
        " nigga",
        " kike",
        " wetback",
        " chink",
        " spic",
        " tranny",
        " faggot",
        " retard",
        " kill yourself",
        " slit your wrists",
        " burn in hell",
        " die in a fire",
        " blowjob",
        " porn",
        " fucking",
        " tramp",
        " boobs",
        " cock",
        " pussy",
        " cocaine",
        " heroin",
        " meth",
        " ecstasy",
        " booze" 
    ]

    offensive = [  " fuck",
        " shit",
        " cunt",
        " asshole",
        " bitch",
        " fucker",
        " wanker"
        ]

    discriminatory = [" nigger",
        " nigga",
        " kike",
        " wetback",
        " chink"]

    violent = [" kill yourself",
        " slit your wrists",
        " burn in hell",
        " die in a fire"]

    sexual = [" blowjob",
        " porn",
        " fucking",
        " tramp",
        " boobs",
        " cock",
        " pussy"]

    drug_alcohol = [" cocaine",
        " heroin",
        " meth",
        " ecstasy",
        " booze"]
    
    #MODEL FOR VIDEO/TEXT ANALYSIS
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=2048)
    
    #OUTPUT FORMAT
    class Output_Format(BaseModel):
        Condition: str = Field(description="""Assigned condition as string. Possible values are "Approved" or "Rejected". """)
        Reason: str = Field(description="""Reason for the assigned condition. Also gives reference from the actual sentence in the context.""" )

    #PARSERS
    json_parser =JsonOutputParser(pydantic_object=Output_Format)
    
    #VIDEO ANALYSIS CHAIN
    chain = (
        RunnablePassthrough()   
        | RunnableLambda(prompt_func)
        | model
        | json_parser
    )
    
    #--------------------------------------------------------
    #BUILD CONTEXT DICTIONARY WHICH CONTAINS IMAGES AND TEXTS
    context_dict = build_context_list(metadatas)
    #print("\n", context_dict)
    #INPUT TEXT
    input_text = "\n".join(list(context_dict["texts"]))
    #INPUT IMAGES LIST
    images = list(context_dict["images"])

    #FORM WARINING AND RESULTS DICTIONARIES FOR STORING DATA
    warnings = {"messages": [], "descriptions": [], "type": []}
    result = {"text": input_text , "data": None, "warning" : {}}

    #START VALIDATION PROCESS
    if  bool(vid_file) == 0:
        warnings["messages"].append("Your post does not contain a video. Only video-based content is eligible for payment. Please add a video to proceed.")
        warnings["descriptions"].append("Video is a requirement for Ganax campaigns. Posts without a video will not be accepted.")
        warnings["type"].append("RED")

    for word in rejected_keywords:
        if word.lower() in input_text:

            keyword_flag = 1

            if word.lower() in offensive:
                warnings["messages"].append("We have not posted your content because it contains offensive language that violates our community guidelines. Please remove any inappropriate terms before resubmitting.")
                warnings["descriptions"].append("The system flags inappropriate or forbidden words from our library of forbidden words. Influencers must remove these words to proceed.")
                warnings["type"].append("RED")
                #print(word.lower())

            elif word.lower() in discriminatory:
                warnings["messages"].append("We have not posted your content as it includes discriminatory language. Kindly ensure all content is respectful and free of offensive references.")
                warnings["descriptions"].append("The system flags inappropriate or forbidden words from our library of forbidden words. Influencers must remove these words to proceed.")
                warnings["type"].append("RED")
                #print(word.lower())

            elif word.lower() in violent:
                warnings["messages"].append("Your content contains language promoting violence or harm, which does not align with our platform policies. Please revise your content accordingly.")
                warnings["descriptions"].append("The system flags inappropriate or forbidden words from our library of forbidden words. Influencers must remove these words to proceed.")
                warnings["type"].append("RED")
                #print(word.lower())

            elif word.lower() in sexual:
                warnings["messages"].append("Your post contains inappropriate sexual content, which is against our community guidelines. Please update your content to meet platform standards.")
                warnings["descriptions"].append("The system flags inappropriate or forbidden words from our library of forbidden words. Influencers must remove these words to proceed.")
                warnings["type"].append("RED")
                #print(word.lower())

            elif word.lower() in drug_alcohol:
        
                warnings["messages"].append("We have not posted your content due to references to drugs, which is prohibited on our platform. Please remove these references before resubmitting.")
                warnings["descriptions"].append("The system flags inappropriate or forbidden words from our library of forbidden words. Influencers must remove these words to proceed.")
                warnings["type"].append("RED")
                print("-----",word)

   
   #PRINT IF THERE ARE WARNINGS OR NOT
    print("Is there warnings?",bool(warnings["messages"]))

    #RUN VIDEO ANALYSIS CHAIN ONLY IF THERE IS NO WARNING BEFORE TO OPTIMIZE COSTS
    if bool(warnings["messages"]) == False:
        print("RUNNING VIDEO APPROVAL CHAIN")
        data = chain.invoke({"context":context_dict})
    else:
        result = {"text": input_text,
                "data": "There are other warnings so we didn't run the video approval model.",
                "warning" : warnings,
                "sentiment" : "There are other warnings so we didn't run the sentiment model."}
        return result

    #-----------------------------------DICTIONARIES TO STORE OUTPUT DATA--------------------------------------------------
    print("CHECKING VIDEO Approval", bool(data))
    #AFTER RUNNING APPROVAL CHAIN IF REJECTED RETURN RESULTS
    if bool(data) != False and data["Condition"] == "Rejected":
        warnings["messages"].append("Your content does not meet the platform’s content guidelines. Please revise the content based on the feedback and resubmit for approval.")
        warnings["descriptions"].append("Influencers must follow detailed branding guidelines. Non-compliant posts will be rejected.")
        warnings["type"].append("RED")
        
        result = {"text": input_text,
                "data": data,
                "warning" : warnings,
                "sentiment" : "There are other warnings so we didn't run the sentiment model."}
        
        return result

    elif bool(data) != False and data["Condition"] == "Approved":

        #ADD WHAT TO DO
        result = {"text": input_text,
                "data": data,
                "warning" : None}

    elif bool(data) == False:
        
        result = {"text": input_text,
                "data": "Approval Model didn't return valid data as expected.",
                "warning" : warnings,
                "sentiment" : "There are other warnings so we didn't run the sentiment model."}
        
        return result

    print("Running sentiment",bool(warnings))
    #RUN SENTIMENT MODEL ONLY IF THERE IS NO WARNING BEFORE TO OPTIMIZE COSTS   
    if bool(warnings["messages"]) == False:
        
        sentiment_out = SentimentModel(r"C:\Users\kerem\Desktop\\Sentiment Model\suerox_comments_rand756.json", [input_text])

        if sentiment_out == "Negative" or sentiment_out == "-1" or sentiment_out == -1:
            warnings["messages"].append("Your post appears speak negatively about the product in the campaign. Please review and adjust your content to ensure a positive sentiment.")
            warnings["descriptions"].append("The system detects negative sentiment about the campaign or product in the content, requiring influencers to modify the post to improve audience reception.")
            warnings["type"].append("RED")
        
            result = {"text": input_text,
                "data": "Rejected",
                "warning" : warnings,
                "sentiment": "Negative"}
            return result

        elif sentiment_out == "Positive" or sentiment_out == "1" or sentiment_out == 1 or sentiment_out == "Neutral" or sentiment_out == "0" or sentiment_out == 0:
            result = {"text": input_text,
                "data": data,
                "warning" : None,
                "sentiment": str(sentiment_out)}
            return result

    else:
        result = {"text": input_text,
                "data": data,
                "warning" : warnings,
                }
        result["sentiment"] = "There are other warnings so we didn't run the sentiment model."
        
        return result

#RUN TEST
vid_url = ""
path = r"C:\Users\kerem\Desktop\Approval_Model\Video_Test_Area\sjh1.MP4"
#vid_dir = r"{}\videos".format(path)
vid_dir = r"C:\Users\kerem\Desktop\Approval_Model\Video_Test_Area\videos"
response = Video_Approval(path, vid_dir)
print("\n", response, "\n")

#SCRIPT ENDED
print("\nScript is working, and ended.")

