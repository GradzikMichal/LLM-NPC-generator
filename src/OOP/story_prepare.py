from pydantic import BaseModel, ConfigDict, Field

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from rich.console import Console

console = Console(force_terminal=True)


class StoryPrepare(BaseModel):
    """
        A class used to prepare story for LLM model.
        :param story_path: Path to the story file.
        :type story_path: str
        :param verbose: Verbose mode. Default is False.
        :type verbose: bool
    """
    model_config = ConfigDict(use_attribute_docstrings=True)
    story_path: str = Field(strict=True, frozen=True, description="Path to the story file.")
    separator: str = Field(strict=True, default="\n", description="Separator for story file.")
    loaded_story: list[Document] = Field(strict=True, default=[], description="Story loaded from file.")
    verbose: bool = Field(strict=True, frozen=True, default=False, description="Verbose mode. Default is False.")

    def __init__(self, story_path: str, verbose: bool) -> None:
        super().__init__(
            story_path=story_path,
            verbose=verbose
        )

    def load_story(self):
        """
            Function for loading a story into a list of Documents.
            :return: Nothing.
            :rtype: None
            :raises error: If file does not exist or file type is not supported.
        """
        file_type: str = self.story_path.split('.')[-1]
        loader = None
        match file_type:
            case "md":
                loader = UnstructuredMarkdownLoader(file_path=self.story_path)
                self.separator = "\n\n"
            case "pdf":
                loader = PyPDFLoader(file_path=self.story_path)
            case "txt":
                loader = TextLoader(file_path=self.story_path)
        if loader is None:
            console.log(f"Error while loading story {self.story_path}. Unrecognized file type.", style="red")
            console.log(f"Exiting the program", style="red")
            exit(1)
        try:
            self.loaded_story: list[Document] = loader.load()
        except FileNotFoundError:
            console.log(f"Error while loading story {self.story_path}. File not found!", style="red")
            console.log(f"Exiting the program", style="red")
            exit(1)
        if self.verbose:
            console.log(f"Story loaded!", style="green")

    def prepare_story(self) -> list[Document]:
        """
            Function for chucking a story.
            :return: List of chunked story.
            :rtype: list[Document]
            :raises error: If user did not load the story using `load_story` method.
        """
        if len(self.loaded_story) == 0:
            console.log("No story loaded! Use load_story method!", style="red")
            exit(1)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator=self.separator)
        documents = text_splitter.split_documents(documents=self.loaded_story)
        if self.verbose:
            console.log(f"Story chunked!", style="green")
        return documents
