# Chat with Multiple PDFs

This project allows users to interact with multiple PDF documents through an intelligent conversational interface. It leverages custom embeddings and a language model (LLM) to process, store, and retrieve information from uploaded PDFs. The application is built with Streamlit for the frontend and LangChain for conversational AI capabilities.

---

## Features
- Upload multiple PDF documents.
- Extract and process text from PDFs.
- Split extracted text into manageable chunks for processing.
- Use custom embeddings (Ollama) for document indexing and retrieval.
- Chat with the uploaded documents using a conversational retrieval chain.
- Persistent chat memory for context-aware interactions.

---

## Technologies Used
- **LangChain**: For building conversational AI and retrieval chains.
- **FAISS**: For efficient vector storage and similarity search.
- **Streamlit**: For creating the interactive web interface.
- **PyPDF2**: For extracting text from PDF files.
- **Ollama**: Custom embeddings and language model (LLM).

---

## Requirements
Install the following dependencies using `pip`:

```bash
pip install langchain==0.0.184
pip install PyPDF2==3.0.1
pip install streamlit==1.18.1
pip install faiss-cpu==1.7.4
pip install altair==4
pip install tiktoken==0.4.0
```

For optional functionality:

```bash
# Uncomment if using HuggingFace LLMs
# pip install huggingface-hub==0.14.1

# Uncomment if using instructor embeddings
# pip install InstructorEmbedding==1.0.1
# pip install sentence-transformers==2.2.2
```

**Additional Requirements:**
- Ollama LLM installed and configured locally on your machine.

---

## Setting Up Ollama Locally

1. **Download and Install Ollama**
   - Visit the [Ollama official website](https://ollama.ai) to download the installer for your operating system.
   - Follow the installation instructions provided on the website.

2. **Download the LLaMA2 Model**
   - Open your terminal or command prompt.
   - Pull the desired model (e.g., LLaMA2) using the command:
     ```bash
     ollama pull llama2
     ```

3. **Start the Ollama Server**
   - Run the following command to start the Ollama server locally:
     ```bash
     ollama serve
     ```
   - Ensure the server is running in the background before starting the application.

---

## How to Run

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Run the Application**
   - Ensure Ollama is installed and running locally.
   - Launch the Streamlit app:
     ```bash
     streamlit run app.py
     ```

3. **Upload PDFs**
   - Use the sidebar to upload multiple PDF files.
   - Click on the "Process" button to extract and process text.

4. **Ask Questions**
   - Type your question in the input field to interact with the content of the uploaded PDFs.

---

## Key Functions

### `get_pdf_text(pdf_docs)`
Extracts text from uploaded PDF documents using PyPDF2.

### `get_text_chunks(text)`
Splits the extracted text into smaller chunks for easier processing using LangChain's `CharacterTextSplitter`.

### `get_vectorstore(text_chunks)`
Creates a FAISS vector store to index text chunks with custom Ollama embeddings.

### `get_conversation_chain(vectorstore)`
Builds a conversational retrieval chain using the custom Ollama LLM and LangChain's memory buffer.

### `handle_userinput(user_question)`
Handles user input, interacts with the conversational chain, and displays the response.

---

## Custom Components

### Ollama Embeddings
A custom embedding class (`OllamaEmbeddings`) that generates embeddings using the local Ollama LLM.

### Ollama LLM Wrapper
A custom language model wrapper (`OllamaLLM`) that integrates the Ollama LLM for prompt-based text generation.

---

## File Structure
```
project-directory/
|-- app.py                  # Main application script
|-- htmlTemplates.py        # HTML templates for styling responses
|-- requirements.txt        # Project dependencies
```

---

## Example Interaction
1. **Upload PDFs:** Upload 2-3 PDFs using the sidebar.
2. **Process PDFs:** Click "Process" to extract and index the text.
3. **Ask Questions:** Example question: "What are the key points from document X?"
4. **View Responses:** Receive detailed responses, dynamically retrieved from the uploaded PDFs.

---

## Known Limitations
- The application requires Ollama installed locally and configured properly.
- Large PDFs may take more time to process.
- Responses depend on the quality of the extracted text from PDFs.

---

## Future Enhancements
- Support for additional document types (e.g., Word, Excel).
- Improved error handling and performance optimization.
- Integration with more LLMs for enhanced conversational abilities.
- Advanced analytics and document summarization capabilities.

---

## Credits
This project uses:
- **LangChain** for conversational AI.
- **FAISS** for vector storage.
- **Ollama** for embeddings and LLM.
- **Streamlit** for the interactive interface.

---

Feel free to contribute to the project or suggest improvements!

