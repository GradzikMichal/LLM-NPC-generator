from typing import Iterator, Sequence
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

import ollama

from langchain_ollama import ChatOllama

console = Console(force_terminal=True)


class LLMModelGenerator(BaseModel):
    """
        A class used to generate Ollama model.
    """
    model_config = ConfigDict(use_attribute_docstrings=True)
    model_name: str = Field(strict=True, frozen=True, description="Name of the Ollama model to generate.")
    model_nargs: list[str] = Field(strict=True, frozen=True,
                                   description="List of additional arguments passed to the model.")
    model_args: dict[str, str | int | float] = Field(strict=True, default_factory=dict[str, str | int | float],
                                                     description="Dictionary of arguments passed to the model.")

    def __init__(self, model_name: str, model_nargs: list[str]) -> None:
        super().__init__(
            model_name=model_name,
            model_nargs=model_nargs
        )

    def generate_model(self) -> ChatOllama:
        """
            Function for generating a model from a model name and  a list of strings.
            :return: Returns a ChatOllama instance
            :rtype: ChatOllama
        """
        if len(self.model_nargs) != 0:
            self.model_args: dict[str, str | int | float] = self._nargs_into_dict()
        if not self._is_model_available():
            self._get_model()
        if self.model_args is not None:
            llm = ChatOllama(model=self.model_name, **self.model_args)
        else:
            llm = ChatOllama(model="llama3.1:8b", num_ctx=10 * 4096, repeat_last_n=-1, repeat_penalty=1.2, )
        console.log(f"Model {self.model_name} created.", style="green")
        return llm

    def _nargs_into_dict(self):
        """
            Function for converting a list of strings into a dictionary
            :rtype: dict[str, str | int | float]
            :return: A dictionary with keys as keys and values as values
            :raises error: If value in a list is not in format of ["key1=value1", "key2=value2", ...]
        """
        try:
            for string in self.model_nargs:
                split_string = string.split('=')
                if split_string[1].isnumeric():
                    self.model_args[split_string[0]] = int(split_string[1])
                elif split_string[1].replace(".", "").isnumeric():
                    self.model_args[split_string[0]] = float(split_string[1])
                else:
                    self.model_args[split_string[0]] = split_string[1]
        except IndexError:
            console.log(f"Error while parsing arguments: {str(self.model_nargs)}. Model arguments in wrong format.",
                        style="red")
            console.log(f"Exiting the program", style="red")
            exit(1)

    def _is_model_available(self) -> bool:
        """
            Function for checking if model is available locally.
            :return: True if model is available locally.
            :rtype: bool
        """
        models: Sequence = ollama.list().models
        models_names: list[str] = []
        for model in models:
            models_names.append(model.model)
        return self.model_name in models_names

    def _get_model(self) -> None:
        """
            Function for getting a specific model from the Ollama database.

            Function taken from examples at https://github.com/ollama/ollama-python/blob/main/examples/pull.py.
        """
        console.log("Model not available locally. Trying to download model.", style="white")
        current_digest, bars = '', {}
        ollama_pull: ollama.ProgressResponse | Iterator[ollama.ProgressResponse] = ollama.pull(self.model_name,
                                                                                               stream=True)
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
