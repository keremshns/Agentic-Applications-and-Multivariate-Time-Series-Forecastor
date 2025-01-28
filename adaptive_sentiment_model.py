import getpass
import os
import bs4
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import json

from pydantic import BaseModel, Field
from enum import Enum
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.enum import EnumOutputParser

import pandas as pd
from operator import itemgetter
import time

#TIME CALCULATIONS
start = time.time()

#API KEY
os.environ["OPENAI_API_KEY"] = "" 
#client = OpenAI()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "" 
#os.environ['USER_AGENT'] = 'myagent'

#CONSTANTS
model_finetuned = ""
model_finetuned_pavo = ""
model_finetuned_suerox = ""  # 
model_instruct = "gpt-3.5-turbo-instruct"
model_chat = "gpt-3.5-turbo"
model_4o_mini = "gpt-4o-mini"
model_4o = "gpt-4o"
model_token_limit = 4096

#FUNCTION DEFINITIONS
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#MODELS
fine_llm = ChatOpenAI(model=model_finetuned_suerox)
llm = ChatOpenAI(model=model_4o_mini)   #OpenAI(gpt-3.5-turbo instruct) or ChatOpenAI (gpt-3.5-turbo)

tesxt_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
#rec_text_split_test = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)

#FINERAG COMMENT OUT FOR RAG DATA
#RAG SOURCE
data_json = []
with open(r"C:\Users\kerem\Desktop\\Data\suerox_comments_rand756.json", "r") as f:
    for line in f:
        if line.strip():  # Skip empty lines
            try:
                data_json.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line}")
                continue
        
print(data_json[0][0]["messages"])

# CONVERT JSON TO TEXT FOR RAG PERFORMANCE
text_l = []
for elem in data_json[0]:
    content = elem["messages"][1]["content"]
    sent = elem["messages"][2]["content"]

    # Prepare text for embedding
    text_to_embed = f""" Content: <{content}> is the given Text Input. Sentiment: <{sent}> is the sentiment of the given text input."""
    text_l.append(text_to_embed)

texts = "\n".join(text_l)
texts = tesxt_splitter.split_text(texts)

num_texts = len(texts)
docs = [Document(page_content=t) for t in texts[:num_texts]]


"""for doc in docs:
    print(doc.page_content)
    print("\n")"""

#FORMING VECTOR DATABASE
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()  


# Define your desired data structure.
class Output_Format(BaseModel):
    Sentiment: str = Field(description="Assigned sentiment as string. Possible values are 1 or -1 or 0")
class Sentiments(Enum):
    Positive = "1"
    Negative = "-1"
    Neutral = "0"


json_parser =JsonOutputParser(pydantic_object=Output_Format)
enum_parser = EnumOutputParser(enum=Sentiments)


#TEMPLATES
fine_rag_template = """You are trained to analyze and detect the sentiment of given text input. Use the following pieces of retrieved context with your training knowledge to answer the question.\n
Analyze the following product review in the user prompt and determine if the sentiment is: Positive, Negative or Neutral. Return answer in single word as either Positive, Negative or Neutral.\n
 
Question: {question} 

Context: {context} 

OUTPUT FORMAT: {format_instructions} 
"""


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
fine_rag_prompt = PromptTemplate(template=fine_rag_template,
                                   input_variables=["question", "context"],
                                   partial_variables={"format_instructions": json_parser.get_format_instructions()})  #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.

rag_prompt = PromptTemplate(template=rag_template,
                                   input_variables=["text_input", "context"]
                                   ).partial(instructions=enum_parser.get_format_instructions())  #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.

fine_prompt = PromptTemplate(template=fine_template,
                                   input_variables=["text"],
                                   partial_variables={"format_instructions": json_parser.get_format_instructions()})  #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.

#TESTING THE MODEL
# Open file 
with open(r"C:\Users\kerem\Desktop\\Data\suerox_comments_lastn_test.json", encoding='utf-8') as f: 

    data = json.load(f)
    #print(data)

         
df = pd.DataFrame(data)
print(df.head())
print(df.shape)
print("\n")
        
#df = df["messages"]
#print(df.head())
df = df.fillna("NaN")
#df = df.tail(128)
#print(df)

#CHAINS
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


human_score = []
model_score = []


for index, row in df.iterrows():   #for index, row in df.iterrows(): LOOP 
    
    row = row["messages"] 
    text = row[1]["content"]
    print(text)
    texts = tesxt_splitter.split_text(text)
    num_texts = len(texts)
    docs = [Document(page_content=t) for t in texts[:num_texts]]
    results = vectorstore.similarity_search_with_relevance_scores(text, k=3)
    context = "\n\n".join([doc.page_content for doc, _score in results])
    #print(docs)
    human_score.append(row[2]["content"])

    if len(results) == 0 or results[0][1] < 0.8:
        #RUN CHAINS
        data = fine_chain.invoke({"docs": docs})
        
        #comparison_output = json.dumps(data, indent=2)
        #print(type(data))
        #print(data)
        
        """
        if type(data) != dict:
            try:
                data = json.loads(data)  # Parse the string to a dictionary
            except:
                print("cant convert to json, data format wont fit json structure")
                try: 
                    data = {"Sentiment": data}
                except:
                    print("thats all I can do for now ;)")"""

        
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
#        try:
#            model_score.append(data["Sentiment"])
#        except:
 #           if type(data) != dict:
  #              try: 
   #                 model_score.append({"Sentiment": data})
    #            except:
    #                model_score.append("NaN")
    #                print("UNEXPECTED OUTPUT FINETUNE OUTPUT IS NOT DICT")
                    #print(data)
     #       else:
     #           model_score.append("NaN")
     #           print("UNEXPECTED OUTPUT FINETUNE (OUTPUT IS DICT)")
                #print(data)


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
            

print("HUMAN SCORE")
print(len(human_score))
print(human_score)
print("MODEL SCORE")
print(len(model_score))
print(model_score)   

end = time.time()
print(end - start)

"""
        try: 
            data = {"Sentiment": data}
        except:
            print("thats all I can do for now")"""