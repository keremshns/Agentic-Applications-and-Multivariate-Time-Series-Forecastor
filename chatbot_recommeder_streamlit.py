import streamlit as st
import os
import time
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions 

from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
#from langchain_chroma import Chroma
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


#API KEY
os.environ["OPENAI_API_KEY"] = ""
#client = OpenAI()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
#os.environ['USER_AGENT'] = 'myagent'

#CONSTANTS
model_finetuned = ""
model_instruct = "gpt-3.5-turbo-instruct"
model_4o = "gpt-4o-mini"
model_35 = "gpt-3.5-turbo"
model_token_limit = 4096
o1mini = "o1-mini"
o1p = "o1-preview"

#Store
store = {}

#FUNCTIONS
def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()

        return store[session_id]

# We want to make sure that we don't have duplicate documents
def remove_duplicates(documents):
    """
    Removes duplicate documents from a list based on their content.

    Parameters:
    documents (list): List of document objects, each with a 'page_content' attribute.

    Returns:
    list: A list of document objects with unique content.
    """
    unique_docs = list({doc.page_content: doc for doc in documents}.values())
    return unique_docs

#INPUT SCHEMAS
class Recommender_Input(BaseModel):
    #rag_path: str = Field(..., description="Path for the Frequently Asked Questions source file for RAG")
    question: str = Field(..., description="""User question which needs to be answered.""")

class QA_EN_Input(BaseModel):
    #rag_path: str = Field(..., description="Path for the Frequently Asked Questions source file for RAG")
    question: str = Field(..., description="User question in english language which needs to be answered.")

class QA_SPA_Input(BaseModel):
    #rag_path: str = Field(..., description="Path for the Frequently Asked Questions source file for RAG")
    question: str = Field(..., description="User question in spanish language which needs to be answered.")

class FAQ_Input(BaseModel):
    #rag_path: str = Field(..., description="Path for the Frequently Asked Questions source file for RAG")
    question: str = Field(..., description="User question which needs to be answered.")

#TOOLS
@tool(args_schema=Recommender_Input)
def Recommender(question: str):
    """Recommends and describes the products that are available in the database and gives suggestions to users according to their question."""

    class Output_Format(BaseModel):
        Answer: str = Field(description="Assistant's answer.")

    llm = ChatOpenAI(model=model_4o, temperature=0.7)   #temperature=0

    docs = []
    for file in os.scandir(r"C:\Users\kerem\Desktop\\Chatbot\Data\Text_Files_Recommendation\Recommedation_Text"):
        
        with open(file, encoding="utf-8") as f:
            read_text = f.read()

        #path = r"C:\Users\kerem\Desktop\Ganax_Local\Chatbot\Data\Files for RAG\Output_Text_Files\FAQ_RAG_Data.txt"

        #recursive text splitter (this might be better to preserve info)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_text(read_text)

        num_texts = len(texts)
        for t in texts[:num_texts]:
            docs.append(Document(page_content=t))

    #data loader
    #loader = CSVLoader(file_path='Data/recommender_content_data1.csv',encoding='utf-8')
    #data = loader.load()

    #print(docs)

    #PARSERS
    json_parser =JsonOutputParser(pydantic_object=Output_Format)
    str_parser = StrOutputParser()

    # Retrieve and generate using the relevant snippets of the blog.
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "score_threshold": 0.5}, fetch__k=num_texts-1, lambda_mult=0.5)

    #similarity search method
    
    #as_retriever(search_type="mmr", search_kwargs={"k": 4})
    #results = vectorstore.max_marginal_relevance_search(str(question), k=4)  #fetch_k=10
    #results = vectorstore.similarity_search_with_relevance_scores(str(question), k=5)
    results = retriever.invoke(question)
    #print(results)

    unique_docs = remove_duplicates(results) 
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    
    #for doc in results:
    #    print("RAG SCORES: ")
    
    prompt_template =  """SYSTEM INSTRUCTION:\n
    You are helpful human assistant who acts as an advisor and responds "Question" of users in a way that they can understand easily\n
    You know both English and Spanish very well. So detect, understand and take action according to the language of the given "Question"\n
    Before providing the answer check the language of the user "Question", than answer with the same language of the user "Question".\n
    Act and response the question like a real human being.\n
    Use only the pieces of retrieved "Context" and please answer the "Question".\n
    Return answer in a single string. Your answer should be clear and concise.\n

    Your job definition is written below:\n
        - Recommending products that are only available in database to users according to their "Question".
        - Don't ever recommend or suggest products which are not in your database !!! 
        - Give suggestions to users, such as which product they may like according to the information which they provide in their "Question".
        - Give suggestions to users, such as which product they may use to replace others according to the information which they provide in their "Question".
        - Describe the products that the user wants according to their "Question".
        - If you don't have information about the product which the user asks in their "Question" at your given "Context" just say "I don't know".   

    USER PROMPT:
    "Question": {question} 

    "Context": {context} 

    "Output Format": {format_instructions}
    
    "Answer":
    """

        
    prompt = PromptTemplate(template=prompt_template,
                                        input_variables=["question", "context"],
                                        partial_variables={"format_instructions": json_parser.get_format_instructions()})

                                        #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.
    #CREATE CHAINS
    llm_chain = ( 
        prompt |
        llm |
        json_parser
        )

    data = llm_chain.invoke({"context": context, "question": question})

    #RUN CHAINS
    return data["Answer"]

@tool(args_schema=QA_EN_Input)
def QA_EN(question: str):
    """Provides useful information and answers user questions which are in english language."""

    #Define your desired output data structure.
    class Output_Format(BaseModel):
        Answer: str = Field(description="Assistant's answer.")
   
    llm = ChatOpenAI(model=model_4o, temperature=0, max_tokens=3072)  

    docs = []
    for file in os.scandir(r"C:\Users\kerem\Desktop\Chatbot\Data\Files for RAG\Blogs_EN\Output_Text_Files"):
        
        with open(file, encoding="utf-8") as f:
            read_text = f.read()

        #path = r"C:\Users\kerem\Desktop\Ganax_Local\Chatbot\Data\Files for RAG\Output_Text_Files\FAQ_RAG_Data.txt"

        #recursive text splitter (this might be better to preserve info)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        texts = text_splitter.split_text(read_text)

        num_texts = len(texts)
        for t in texts[:num_texts]:
            docs.append(Document(page_content=t))
    
    for file in os.scandir(r"C:\Users\kerem\Desktop\\Chatbot\Data\Files for RAG\FAQ\Output_Text_Files"):
        
        with open(file, encoding="utf-8") as f:
            read_text = f.read()

        #path = r"C:\Users\kerem\Desktop\Ganax_Local\Chatbot\Data\Files for RAG\Output_Text_Files\FAQ_RAG_Data.txt"

        #recursive text splitter (this might be better to preserve info)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        texts = text_splitter.split_text(read_text)

        num_texts = len(texts)
        for t in texts[:num_texts]:
            docs.append(Document(page_content=t)) 

    # Retrieve and generate using the relevant snippets of the blog.
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    #retriever = vectorstore.as_retriever()

    #similarity search method
    #retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.5})
    results = vectorstore.similarity_search_with_relevance_scores(str(question), k=4, score_threshold=0.5)
    context = "\n\n".join([doc.page_content for doc, _score in results])

    print(results)
    #results = retriever.invoke(question)
    #unique_docs = remove_duplicates(results) 
    #context = "\n\n".join([doc.page_content for doc in results])
   
    #print("CONTEXT FOUND IN THE FILE: ", docs)

    json_parser =JsonOutputParser(pydantic_object=Output_Format)
    str_parser = StrOutputParser()
    
    prompt_template =  """SYSTEM INSTRUCTION:\n
    You are helpful human assistant who acts as an advisor and responds questions of users in a way that they can understand easily\n
    You know both English and Spanish very well. So detect, understand and take action according to the language of the given "Question"\n
    Before providing the answer check the language of the user "Question", than answer with the same language of the user "Question".\n
    Act and response the question like a real human being.\n
    Just use information available in the "Context".\n

    USER PROMPT:
    Your job is to analyze and understand the user question. Then using the following pieces of retrieved context please answer the question.\n
    Return answer in a single string. Your answer should be clear and concise. Don't make up content or come up with answers that does not exist in the context.\n
    If you don't know the answer, just tell.\n
    
    "Question": {question} 

    Context: {context} 

    Output Format: {format_instructions}
    
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables = ["question", "context"],
                            partial_variables = {"format_instructions":json_parser.get_format_instructions()})
                             #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.
    #CREATE CHAINS
    rag_chain = ( 
    prompt |
    llm |
    json_parser
    )

    data = rag_chain.invoke({"context": context, "question": str(question)})
    #comparison_output = json.dumps(data, indent=2)
    #print(type(data))
    #print(data)

    #RUN CHAINS
    return data["Answer"]


@tool(args_schema=QA_SPA_Input)
def QA_SPA(question: str):
    """Provides useful information and explain related user questions which are in spanish language."""

    #Define your desired output data structure.
    class Output_Format(BaseModel):
        Answer: str = Field(description="Assistant's answer.")
   
    llm = ChatOpenAI(model=model_4o, temperature=0, max_tokens=3072)  

    docs = []
    for file in os.scandir(r"C:\Users\kerem\Desktop\\Chatbot\Data\Files for RAG\Blogs_SPA\Output_Text_Files"):
        
        with open(file, encoding="utf-8") as f:
            read_text = f.read()

             #path = r"C:\Users\kerem\Desktop\Ganax_Local\Chatbot\Data\Files for RAG\Output_Text_Files\FAQ_RAG_Data.txt"

        #recursive text splitter (this might be better to preserve info)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        texts = text_splitter.split_text(read_text)

        num_texts = len(texts)
        for t in texts[:num_texts]:
            docs.append(Document(page_content=t))
    
    for file in os.scandir(r"C:\Users\kerem\Desktop\Chatbot\Data\Files for RAG\FAQ\Output_Text_Files"):
        
        with open(file, encoding="utf-8") as f:
            read_text = f.read()

        #path = r"C:\Users\kerem\Desktop\Ganax_Local\Chatbot\Data\Files for RAG\Output_Text_Files\FAQ_RAG_Data.txt"

        #recursive text splitter (this might be better to preserve info)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        texts = text_splitter.split_text(read_text)

        num_texts = len(texts)
        for t in texts[:num_texts]:
            docs.append(Document(page_content=t)) 

    # Retrieve and generate using the relevant snippets of the blog.
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    #retriever = vectorstore.as_retriever()

    #similarity search method
    #retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.5})
    results = vectorstore.similarity_search_with_relevance_scores(str(question), k=4, score_threshold=0.5)
    context = "\n\n".join([doc.page_content for doc, _score in results])

    print(results)
    #results = retriever.invoke(question)
    #unique_docs = remove_duplicates(results) 
    #context = "\n\n".join([doc.page_content for doc in results])
   
    #print("CONTEXT FOUND IN THE FILE: ", docs)

    json_parser =JsonOutputParser(pydantic_object=Output_Format)
    str_parser = StrOutputParser()
    
    prompt_template =  """SYSTEM INSTRUCTION:\n
    You are helpful human assistant who acts as an advisor and responds questions of users in a way that they can understand easily\n
    You know both English and Spanish very well. So detect, understand and take action according to the language of the given "Question"\n
    Before providing the answer check the language of the user "Question", than answer with the same language of the user "Question".\n
    Act and response the question like a real human being.\n
    Just use information available in the "Context".\n

    USER PROMPT:
    Your job is to analyze and understand the user question. Then using the following pieces of retrieved context please answer the question.\n
    Return answer in a single string. Your answer should be clear and concise. Don't make up content or come up with answers that does not exist in the context.\n
    If you don't know the answer, just tell.\n
    
    "Question": {question} 

    Context: {context} 

    Output Format: {format_instructions}
    
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables = ["question", "context"],
                            partial_variables = {"format_instructions":json_parser.get_format_instructions()})
                             #This is a prompt for retrieval-augmented-generation. It is useful for chat, QA, or other applications that rely on passing context to an LLM.
    #CREATE CHAINS
    rag_chain = ( 
    prompt |
    llm |
    json_parser
    )

    data = rag_chain.invoke({"context": context, "question": str(question)})
    #comparison_output = json.dumps(data, indent=2)
    #print(type(data))
    #print(data)

    #RUN CHAINS
    return data["Answer"]


#FORMING AGENT AND CHAINS

#REQUIRED TOOLS LIST FOR MODEL TO USE / ONLY SELECT THE USEFUL ONES
tools = [Recommender, QA_EN, QA_SPA]

#Convert tools to openai functions and bind them to model
functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(model_name=model_4o, temperature=0).bind(functions=functions)

#Chat Prompt Template with chat memory and tool response memory
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ganax is a platform that connects influencers with brands, providing tools for campaign creation, collaboration, and monetization. We offer training, brand collaborations, and personalized content creation support.\n
     You are the helpful human assistant of Ganax who responds to questions of users in a way that they can understand easily. Your name is "Gigi".\n
     Please greet the users only in your first response, let them know who you are and tell them you are there to help answer their questions. Act and response the question like a real human being.\n
     """),
    #MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

#Creating base chain, agent chain () and memory
chain = prompt | model | OpenAIFunctionsAgentOutputParser() 

agent_chain = RunnablePassthrough.assign(agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])) | chain        

#memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")  #NOT USING SINCE IM USING CHUONGS METHOD TO GET HISTORY
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

test = """messages = []
prompt = "Can you recommend some products"
messages.append({"role": "user", "content": prompt})
# Get agent response from langchain !!!
prompt_w_hist = str([ {"role": m["role"], "content": m["content"]} for m in messages])

agent_response = agent_executor.invoke({"input": prompt_w_hist},
                                     #config={"configurable": {"session_id": str(st.context.cookies.to_dict().get("ajs_anonymous_id"))}}
                                     )
messages.append({"role": "assistant", "content": agent_response["output"]})
print(agent_response["output"])"""

#AGENT EXECUTOR WITH HISTORY   
a = """agent_executor = RunnableWithMessageHistory(
        agent_executor_main,
        get_session_history,        
        input_messages_key="input",
        history_messages_key="chat_history",
        history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True)]
    )"""
    
#TEST UI WITH STREAMLIT
recommender_question = """Suggest 3 anime names which are similar to Cowboy Bebop.
                        Follow this approach 1: Find similar Animes using context and 2: suggest new similar animes.
                        Also give a reason for suggestion"""

# Streamed response emulator
def response_generator(text):
    
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

st.title("GANAX CHATBOT with Recommender, FAQ and Guide")

# Initialize chat history

if 'messages' not in st.session_state:
    st.session_state.messages = []
   
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar="😎"):
            st.markdown(":orange[USER:]\n")
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="🤖"):
            st.markdown(":blue[ASSISTANT:]\n")
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Get agent response from langchain !!!
    
    prompt_w_hist = str([ {"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]] )

    agent_response = agent_executor.invoke({"input": prompt_w_hist},
                                     #config={"configurable": {"session_id": str(st.context.cookies.to_dict().get("ajs_anonymous_id"))}}
                                     )
    

    #print("RAW MODEL RESPONSE:\n", agent_response["output"])    #FOR TERMINAL
    #agent_response = agent_executor.invoke({"input": prompt})

    #print(st.context.cookies)
    #print(st.context.cookies.to_dict())

    # Display user message in chat message container
    with st.chat_message("user", avatar="😎"):

        generator_response = response_generator(prompt)
        st.markdown(":orange[USER:]\n")
        st.write_stream(generator_response)
        

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": agent_response["output"]})

    # Display assistant message in chat message container
    with st.chat_message("assistant", avatar="🤖"): 

        #print("STORE IS WORKING OUTSIDE: ", store)
        generator_response = response_generator(agent_response["output"])
        st.markdown(":blue[ASSISTANT:]\n")
        st.write_stream(generator_response)
    
    


