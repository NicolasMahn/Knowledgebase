from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

from query_data import PROMPT_TEMPLATE

def query(context: str, question: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    question = """
    Given the context, what further information sources would be relevant to answer questions about the Franka Emika 
    Panda Robot, and how to run it? 
    Please give a short answer in two segments. Explain what the context is about first, and then what you would like 
    to know on top of the context about the Robot. You answer should be as short as possible.
    """
    prompt = prompt_template.format(context=context, question=question)

    model = Ollama(model="mistral")
    response = model.invoke(prompt)

    print(response)

    return response


def read(text: str):

    question = """
    Given the context, what further information sources would be relevant to answer questions about the Franka Emika 
    Panda Robot, and how to run it? 
    Please give a short answer in two segments. Explain what the context is about first, and then what you would like 
    to know on top of the context about the Robot. You answer should be as short as possible.
    """

    return query(text, question)


def aggregate(insights: str):

    question = """
    The context comprises expert analysis of datasource's surrounding the Franka Emika Panda Robot. 
    Please summarize what information sources are missing. 
    I. e. which source have not been mentioned by any experts but have been requested.
    """

    return query(insights, question)