# Why : 
https://medium.com/@harshil00raval/llm-with-rag-conversation-with-ai-45fc09231727

I was constantly hitting Context size (No of tokens (chatGPT : 8k to 32k , claude : 200K)) limits with GenAI chats 
where I was dealing with code bases. The context size is a real speed limiter.
I was exploring multiple things to bypass that when RAG hit my questioning streak.

Unfortunately, RAG is not a solution to Context size problem but the knowledge gained from those LLM interaction was 
worth documenting.

"RAG does NOT help increase the LLM’s context window.RAG just automates amount of info developer has to pass as 
context to the chat. Context = Documentation + query. RAG enables LLM to read Documentation directly from the 
Vector databases."

In this project I am trying to create a RAg pipeline as A POC.

Added Features : 
- Document support
- Configure the data using an API in realtime
- remove the data using an API in realtime

# Build :
docker build -t faiss-service .

# Run : Yu have to mount a directory from where you want to select a directory to upload for encoding
docker run -p 8000:8000 -v ./documents:/app/documents faiss-service
$ docker run -p 8000:8000 -v /home/harshil/Development:/app/documents faiss-service

# Usage :

## Add Files : 
curl -X POST "http://localhost:8000/add_files/" -H "Content-Type: application/json" -d '{"directory": "/app/documents/RAGforLLM/documents"}'
## response :
{"message":"Added 1 documents to FAISS"}

## Remove Files :
curl -X POST "http://localhost:8000/remove_file/" -H "Content-Type: application/json" -d '{"file_name": "harshil.txt"}'
## response : 
{"message":"Removed harshil.txt from FAISS"}

## List Files :
curl -X GET "http://localhost:8000/list_files/"
## response :
{"message":"No files are currently encoded in FAISS."}
or
{"encoded_files":[{"file_name":"README.md","index":0},{"file_name":"requirements.txt","index":1}]}

## Query :
curl "http://localhost:8000/query/?query=How%20to%20monitor%20a%20Java%20app%3F"
# response :
{"document":"Monitoring with Prometheus"}   