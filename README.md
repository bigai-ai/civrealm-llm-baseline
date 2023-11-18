# civ-LLMs

The llm baseline part for freeciv env.

USAGE:
(1) Install civrealm

(2) Set env varibles.

```
# Use AZURE_OPENAI_API_TYPE="azure" to use Azure LLM, otherwise use "openai"
export AZURE_OPENAI_API_TYPE="<your_open_api_type>"
export AZURE_OPENAI_API_VERSION='<your_openai_api_version>'
export AZURE_OPENAI_API_BASE='<your_openai_api_base>'
export AZURE_OPENAI_API_KEY='<your_openai_api_key>'
export LOCAL_LLM_URL='<if_need_local_llm_inference>'
export MY_PINECONE_API_KEY='<your_pinecone_api_key>'
export MY_PINECONE_ENV='<your_pinecone_env_name>'
```

(3) Execute the code.
`python main.py`
