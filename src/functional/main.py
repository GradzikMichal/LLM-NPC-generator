import argparse
from typing import Iterator
from rich.console import Console
from tqdm import tqdm
import torch
import ollama
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser

console = Console(force_terminal=True)


def nargs_into_dict(list_of_str: list[str]) -> dict[str, str | int | float]:
    """
        Function for converting a list of strings into a dictionary
        :param list_of_str: List of strings in format of ["key1=value1", "key2=value2", ...]
        :type list_of_str: list[str]
        :rtype: dict[str, str | int | float]
        :return: A dictionary with keys as keys and values as values
        :raises error: If value in a list is not in format of ["key1=value1", "key2=value2", ...]
    """
    arg_dict: dict[str, str | int | float] = {}
    try:
        for string in list_of_str:
            split_string = string.split('=')
            if split_string[1].isnumeric():
                arg_dict[split_string[0]] = int(split_string[1])
            elif split_string[1].replace(".", "").isnumeric():
                arg_dict[split_string[0]] = float(split_string[1])
            else:
                arg_dict[split_string[0]] = split_string[1]
        return arg_dict
    except IndexError:
        console.log(f"Error while parsing arguments: {str(list_of_str)}. Model arguments in wrong format.",
                    style="red")
        console.log(f"Exiting the program", style="red")
        exit(1)


def get_available_models() -> list[str]:
    """
        Function for getting names of locally available models.
        :return: A list of available models names.
        :rtype: list[str]
    """
    models: list = ollama.list().models
    models_names: list[str] = []
    for model in models:
        models_names.append(model.model)
    return models_names


def get_model(model_name: str) -> None:
    """
        Function for getting a specific model from the Ollama database.
        :param model_name: Name of the model to create. List of available models at https://ollama.com/models.
        :type model_name: str

        Function taken from examples at https://github.com/ollama/ollama-python/blob/main/examples/pull.py.
    """
    console.log("Model not available locally. Trying to download model.", style="white")
    current_digest, bars = '', {}
    ollama_pull: ollama.ProgressResponse | Iterator[ollama.ProgressResponse] = ollama.pull(model_name, stream=True)
    for response_progress in ollama_pull:
        digest = response_progress.get('digest', '')
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()
        if not digest:
            console.print(response_progress.get('status'))
            continue
        if digest not in bars and (total := response_progress.get('total')):
            bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)
        if completed := response_progress.get('completed'):
            bars[digest].update(completed - bars[digest].n)
        current_digest = digest
    console.log("Download complete!", style="green")


def model_creator(model_name: str, model_nargs: list[str]) -> ChatOllama:
    """
    Function for creating a model from a model name and  a list of strings.
    :param model_name: Name of the model to create. List of available models at https://ollama.com/models.
    :type model_name: str
    :param model_nargs: List of strings in format of ["key1=value1", "key2=value2", ...]. Each string is an argument and its value of a model.
    :type model_nargs: q
    :return: Returns a ChatOllama instance
    :rtype: ChatOllama
    """
    model_args: None | dict = None
    if len(model_nargs) != 0:
        model_args = nargs_into_dict(model_nargs)
    available_models: list[str] = get_available_models()
    if model_name not in available_models:
        get_model(model_name)
    if model_args is not None:
        llm = ChatOllama(model=model_name, **model_args)
    else:
        llm = ChatOllama(model="llama3.1:8b", num_ctx=10 * 4096, repeat_last_n=-1, repeat_penalty=1.2, )
    console.log(f"Model {model_name} created.", style="green")
    return llm


def load_story(story_path: str, verbose: bool) -> (list[Document], str):
    """
        Function for loading a story into a list of Documents.
        :param story_path: Path to the story file. Supported file types include: [.txt, .md, .pdf].
        :type story_path: str
        :param verbose: Verbose mode. Boolean to enable verbose mode. Default is False. Optional
        :type verbose: bool
        :return: Story as list of Document and separator.
        :rtype: (list[Document], str)
        :raises error: If file does not exist or file type is not supported.
    """
    file_type: str = story_path.split('.')[-1]
    loader = None
    sep: str = '\n'
    match file_type:
        case "md":
            loader = UnstructuredMarkdownLoader(file_path=story_path)
            sep = "\n\n"
        case "pdf":
            loader = PyPDFLoader(file_path=story_path)
        case "txt":
            loader = TextLoader(file_path=story_path)
    if loader is None:
        console.log(f"Error while loading story {story_path}. Unrecognized file type.", style="red")
        console.log(f"Exiting the program", style="red")
        exit(1)
    try:
        loaded_story: list[Document] = loader.load()
    except FileNotFoundError:
        console.log(f"Error while loading story {story_path}. File not found!", style="red")
        console.log(f"Exiting the program", style="red")
        exit(1)
    if verbose:
        console.log(f"Story loaded!", style="green")
    return loaded_story, sep


def prepare_story(loaded_story: list[Document], sep: str, verbose: bool) -> list[Document]:
    """
        Function for chucking a story.
        :param loaded_story: Loaded story file.
        :type loaded_story: list[Document]
        :param sep: Separator for story file.
        :type sep: str
        :param verbose: Verbose mode. Boolean to enable verbose mode. Default is False. Optional
        :type verbose: bool
        :return: List of chunked story.
        :rtype: list[Document]
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator=sep)
    documents = text_splitter.split_documents(documents=loaded_story)
    if verbose:
        console.log(f"Story chunked!", style="green")
    return documents


def prepare_vectorstore(chunked_story: list[Document], rag_path: str, verbose: bool) -> Chroma:
    """
        Function for creating a vector store which enables RAG functionality. Uses Chroma as a vector store.
        :param chunked_story: Chunked story
        :type chunked_story: list[Document]
        :param rag_path: Path to the RAG folder. If provided RAG becomes persisted and database is saved in provided folder.
        :type rag_path:
        :param verbose: Verbose mode. Boolean to enable verbose mode. Default is False. Optional
        :type verbose: bool
        :return: Returns a vector store. Right now vectorstore is not persistent.
        :rtype: Chroma
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    if rag_path == "":
        vectorstore = Chroma().from_documents(chunked_story, embeddings)
    else:
        vectorstore = Chroma(persist_directory=rag_path).from_documents(chunked_story, embeddings)
    if verbose:
        console.log(f"Vectorstore created!", style="green")
    return vectorstore


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


# model name, history file
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Interactive local NPCs generator with parser",
        usage="%(prog)s [options] --model_name"
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
    llm_model = model_creator(args.model_name, args.model_args)
    story, separator = load_story(args.story_path, args.verbose)
    story = prepare_story(story, separator, args.verbose)
    rag = prepare_vectorstore(story, args.rag_folder, args.verbose)
    prompt_template = prompt_template()
    OllamaLLM = combining_llm_with_rag(llm=llm_model, prompt=prompt_template, vectorstore=rag)
    console.log("LLM ready to use! Have fun!", style="green")
    while True:
        user_input = input("Provide prompt for a model or write quit to exit: ")
        if user_input.lower() == "quit":
            console.log(f"Exiting the program.", style="red")
            exit(0)
        result = OllamaLLM.invoke(user_input)
        print(result)
