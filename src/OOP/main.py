import argparse
from rich.console import Console

from story_prepare import StoryPrepare
from vectorstore_creator import VectorStoreCreator
from model_generator import LLMModelGenerator
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser

console = Console(force_terminal=True)


def prompt_template() -> PromptTemplate:
    """
        Function which is used to create the prompt template.
        :return: Returns a prompt template.
        :rtype: PromptTemplate
    """
    prompt: str = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n"
        "Only when user ask you to generate a NPC or a character then you must to give an answer in JSON format, user will provide what he wants. "
        "If user ask for traits you must return list of traits using only one word for each trait. Do not explain this JSON. \n"
        "If user ask for NPC name you must generate new and unique NPC name each time. \n"
        "In any other case return normal answer if you don't know the answer, just say that you don't know.\n"
        "Question: {question}"
        "Context of possible questions from user: {context}")
    template: PromptTemplate = PromptTemplate(
        input_variables=["question", "context"], template=prompt
    )
    return template


def combining_llm_with_rag(llm: ChatOllama, prompt: PromptTemplate, vectorstore: Chroma) -> RunnableSerializable:
    """
        Function for combining LLM with RAG and prompt template.
        :param llm: LLM model created using `model_creator()`
        :type llm: ChatOllama
        :param prompt: Prompt template created using `prompt_template()`
        :type prompt: PromptTemplate
        :param vectorstore: Vectorstore created using `prepare_vectorstore()`
        :type vectorstore: Chroma
        :return: Returns prepared LLM with RAG and prompt template.
        :rtype: RunnableSerializable
    """
    rag_llm = ({
                   "context": vectorstore.as_retriever(search_kwargs={"k": 10}),
                   "question": RunnablePassthrough(),
               }
               | prompt
               | llm
               | StrOutputParser()
               )
    return rag_llm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Interactive local NPCs generator with parser",
        usage="%(prog)s [options] --model_name",
    )
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1:8b",
                        help="Name of local Ollama LLM model to use. Default is llama3.1:8b")
    parser.add_argument("--model_args", nargs="*", default=[],
                        help="Additional arguments of the model. Use in a format key=value")
    parser.add_argument("-s", "--story_path", required=True, type=str, help="Path to the story file.")
    parser.add_argument("-r", "--rag_folder", default="", type=str,
                        help="Provide path for a RAG folder. Default is chroma.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose mode.")
    args = parser.parse_args()
    story_prepare = StoryPrepare(story_path=args.story_path, verbose=args.verbose)
    story_prepare.load_story()
    story = story_prepare.prepare_story()
    RAG = VectorStoreCreator(rag_path=args.rag_folder, verbose=args.verbose).create_vectorstore(story)
    llm_model = LLMModelGenerator(model_name=args.model_name, model_nargs=args.model_args).generate_model()
    prompt_template = prompt_template()
    OllamaLLM = combining_llm_with_rag(llm=llm_model, prompt=prompt_template, vectorstore=RAG)
    console.log("LLM ready to use! Have fun!", style="green")
    while True:
        user_input = input("Provide prompt for a model or write quit to exit: ")
        if user_input.lower() == "quit":
            console.log(f"Exiting the program.", style="red")
            exit(0)
        result = OllamaLLM.invoke(user_input)
        print(result)
