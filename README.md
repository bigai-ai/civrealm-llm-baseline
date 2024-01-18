# BaseLang & Mastaba: Civrealm-LLM-Agents
BaseLang and Mastaba are two LLM-based agents for the reinforcement learning environment [CivRealm](https://www.github.com/bigai-ai/civrealm). BaseLand and Mastaba share a similar interface to CivRealm over standard Gymnasium, as provided in CivRealm. BaseLang implements a paraallel controller on each unit individually and Mastaba uses an "advisor" to lead them all. Both agent models are presented in the paper of CivRealm.

## Prerequisit
`civrealm` from [CivRealm](https://www.github.com/bigai-ai/civrealm).
The list `requirements.txt` in the repository.

## USAGE:
1. Install civrealm properly with correct freeciv-web. (See [CivRealm](https://www.github.com/bigai-ai/civrealm))

2. Prepare the LLM's to use (GPT api key or local LLM URL)

3. Prepare a `PINECONE` API Key.

4. Set env varibles.

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

5. Execute the code.
`python main.py`
