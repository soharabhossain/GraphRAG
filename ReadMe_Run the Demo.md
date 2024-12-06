--------------------------------------------
Step-by-Step Instructions to Run the Demo
--------------------------------------------

-----------------
Execute The Demo
-----------------

Open a terminal in your project folder and run the following commands (assume Python 3.10 is already installed in your system).

1. Create a viraual environment
python -m  venv graphrag
----------------------------------
2. Activate the virtual env
source graphrag/bin/activate
or
graphrag/Scripts/activate
----------------------------------
3. Install Dependencies
pip install graphrag
-------------------------------------------------------------

4. Set up a data project and some initial configuration. 
First let's get a sample dataset ready:

Create a folder "ragtest".
Create a subfolder inside it named "input".
Put a sample text file inside the "ragtest/input" folder

-------------------------------------------------------------

5. Set Up Your Workspace Variables

To initialize your workspace, first run the graphrag init command. Since we have already configured a directory named ./ragtest in the previous step, run the following command:

graphrag init --root ./ragtest


Inside the ragtest directory
This will create two files: .env and settings.yaml in the ./ragtest directory.

.env contains the environment variables required to run the GraphRAG pipeline. If you inspect the file, you'll see a single environment variable defined, GRAPHRAG_API_KEY=<API_KEY>. This is the API key for the OpenAI API or Azure OpenAI endpoint. You can replace this with your own API key. If you are using another form of authentication (i.e. managed identity), please delete this file.

settings.yaml contains the settings for the pipeline. You can modify this file to change the settings for the pipeline.

Also, a folder named prompts will be created with a bunch of text files with different prompts for performing different tasks (e.g., entity extraction, local/global search, summarization, etc.)

-------------------------------------------------------------
Step: 5.1 [Run with OpenAI]
Set the OpenAI API Key
----------------------
Goto the .env file and set the API key as EMPTY as follows:
GRAPHRAG_API_KEY=<place your OpenAi API key here>, save the file.

Alternatively use Ollama to run the LLM and the enbedding models locally (check the instructions below, Step 5.1: [Alternative]).

-------------------------------------------------------------

6. Running the Indexing pipeline
Finally we'll run the pipeline!

graphrag index --root ./ragtest


This process will take some time to run. 
This depends on the size of your input data, what model you're using, and the text chunk size being used (these can be configured in your settings.yml file). 

This will process the text/document - chunk it, find embeddings for each chunk and store in a local vector database (LanceDB).


Once the pipeline is complete, you should see a new folder called ./ragtest/output with a series of parquet files.

----------------------------------

7. Run the query Engine

graphrag query --root ./ragtest --method local --query "Who is David Copperfield and what are his main relationships?"

graphrag query --root ./ragtest --method global --query "What is the overall theme of the story?"


graphrag query --root ./ragtest --method local --query "How does YOLO use grid division to predict object locations and classifications in an image?"

graphrag query --root ./ragtest --method global --query What are the key features that make YOLO efficient for real-time object detection tasks?


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Step 5.1: [Alternative] Use local LLMs with Ollama
--------------------------------------------------

i. Download and Install Ollama

ii. Install the ollama sdk withing the "graphrag" virtual env.
pip install ollama
	
iii. Pull two models from Ollama model repository/hub:
Run the following commands on a terminal.

Download the LLM: 
ollama pull llama3.1

Download the Embedding model: 
ollama pull nomic-embed-text


Update settings.yaml
Modify these changes in settings.yaml to use Ollama:

llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  api_base: http://localhost:11434/v1 # local address for llama3.1 


Within "embeddings:" update the llm section as follows:

  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: nomic-embed-text
    api_base: http://localhost:11434/api # local address for the embedding model

------------------------------------------

Goto the installation directory of your virtual environment
graphrag/Lib/site-packages/graphrag

graphrag/llm/openai/openai_embeddings_llm.py
Replace the context of this file with the following code and save the file.

from typing_extensions import Unpack
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
import ollama

class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self._client = client
        self._configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self._configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        embedding_list = []
        for inp in input:
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=inp)
            embedding_list.append(embedding["embedding"])
        return embedding_list


------------------------------------------
Goto the .env file and set the API key as EMPTY as follows:
GRAPHRAG_API_KEY=EMPTY

-----------------------------------------------------------------

