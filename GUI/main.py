import sys

from rich.console import Console
from src.OOP.model_generator import LLMModelGenerator
from src.OOP.story_prepare import StoryPrepare
from src.OOP.vectorstore_creator import VectorStoreCreator
from src.functional.main import get_available_models, prompt_template, combining_llm_with_rag
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QCheckBox, QDialog, QGridLayout,
                               QLineEdit, QPushButton, QSizePolicy, QTextBrowser, QFileDialog, QListWidget)

console = Console(force_terminal=True)


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.Signal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


def run_llm_model(model_name, model_nargs, story_path, rag_folder, verbose):
    story_prepare = StoryPrepare(story_path=story_path, verbose=verbose)
    story_prepare.load_story()
    story = story_prepare.prepare_story()
    RAG = VectorStoreCreator(rag_path=rag_folder, verbose=verbose).create_vectorstore(story)
    llm_model = LLMModelGenerator(model_name=model_name, model_nargs=model_nargs).generate_model()
    prompt_temp = prompt_template()
    OllamaLLM = combining_llm_with_rag(llm=llm_model, prompt=prompt_temp, vectorstore=RAG)
    console.log("LLM ready to use! Have fun!", style="green")
    return OllamaLLM


class MyWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.available_models = get_available_models()

        self.initUI()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def initUI(self):
        self.setWindowTitle('LLM for NPC generation')
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(50, 50, 50, 50)
        self.setLayout(self.main_layout)
        self.set_grid_columns()
        self.file_layout()
        self.get_model_name()
        self.model_additional_args()
        self.rag_folder()
        self.verbose_check()
        self.start_model_button()
        self.chat_window()
        self.prompt_input()

    def set_grid_columns(self):
        text1 = QTextBrowser(self)

        text2 = QTextBrowser(self)
        text3 = QTextBrowser(self)
        text4 = QTextBrowser(self)
        text1.setMaximumSize(QSize(10, 10))
        text2.setMaximumSize(QSize(10, 10))
        text3.setMaximumSize(QSize(10, 10))
        text4.setMaximumSize(QSize(10, 10))
        self.main_layout.addWidget(text1, 0, 0)
        self.main_layout.addWidget(text2, 0, 1)
        self.main_layout.addWidget(text3, 0, 2)
        self.main_layout.addWidget(text4, 0, 3)

    def file_layout(self):
        self.text_dir_widget = QTextBrowser(self)
        self.text_dir_widget.setPlainText("Please select a story file")
        self.text_dir_widget.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.text_dir_widget.setFixedHeight(30)
        dirButton = QPushButton('Browse')
        dirButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dirButton.clicked.connect(self.openFileDialog)
        dirButton.setFixedHeight(30)
        self.main_layout.addWidget(self.text_dir_widget, 1, 0, 1, 3)
        self.main_layout.addWidget(dirButton, 1, 3)

    def get_model_name(self):
        self.model_text_edit = QLineEdit(self)
        self.model_text_edit.setPlaceholderText("Please write a model name")
        self.listWidget = QListWidget()
        self.listWidget.addItems(self.available_models)
        self.listWidget.clicked.connect(self.model_list_clicked)
        self.listWidget.setFixedHeight(100)
        self.main_layout.addWidget(self.listWidget, 2, 0, 1, 1)
        self.main_layout.addWidget(self.model_text_edit, 2, 1, 1, 3)

    def model_additional_args(self):
        self.model_additional_args_list = QLineEdit(self)
        self.model_additional_args_list.setPlaceholderText(
            "Optionally write an additional argument in format key1=value1 key2=value2")
        self.main_layout.addWidget(self.model_additional_args_list, 3, 0, 1, 4)

    def rag_folder(self):
        self.rag_dir_widget = QTextBrowser(self)
        self.rag_dir_widget.setPlainText("Please select a RAG directory")
        self.rag_dir_widget.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.rag_dir_widget.setFixedHeight(30)
        dirButton = QPushButton('Browse')
        dirButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dirButton.clicked.connect(self.openDirDialog)
        dirButton.setFixedHeight(30)
        self.main_layout.addWidget(self.rag_dir_widget, 4, 0, 1, 3)
        self.main_layout.addWidget(dirButton, 4, 3)

    def verbose_check(self):
        self.verbose_bool = QCheckBox("Verbose?")
        self.main_layout.addWidget(self.verbose_bool, 5, 1, 1, 1)

    def start_model_button(self):
        start_button = QPushButton('Start model')
        start_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        start_button.clicked.connect(self.start_model)
        start_button.setFixedHeight(30)
        self.main_layout.addWidget(start_button, 5, 2, 1, 1)

    def chat_window(self):
        self.chat_widget = QTextBrowser(self)
        self.chat_widget.setPlainText("Waiting for model to be started")
        self.chat_widget.setAcceptRichText(True)
        self.chat_widget.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.main_layout.addWidget(self.chat_widget, 6, 0, 3, 4)

    def prompt_input(self):
        self.prompt_send = QLineEdit(self)
        self.prompt_send.setPlaceholderText("Please write a prompt")
        send_button = QPushButton('Send prompt')
        send_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        send_button.clicked.connect(self.send_prompt)
        send_button.setFixedHeight(30)
        self.main_layout.addWidget(self.prompt_send, 10, 0, 1, 3)
        self.main_layout.addWidget(send_button, 10, 3, 1, 1)

    def openFileDialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Choose story file")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            self.text_dir_widget.setPlainText(selected_files[0])

    def openDirDialog(self):
        dir_dialog = QFileDialog(self).getExistingDirectory(self)
        self.rag_dir_widget.clear()
        self.rag_dir_widget.setPlainText(dir_dialog)

    def start_model(self):
        story_path = self.text_dir_widget.toPlainText()
        model_name = self.model_text_edit.text()
        model_nargs = self.model_additional_args_list.text()
        if model_nargs == '':
            model_nargs = []
        else:
            model_nargs = model_nargs.split(' ')
        rag_dir = self.rag_dir_widget.toPlainText()
        verbose = self.verbose_bool.isChecked()
        try:
            self.llm_model = run_llm_model(model_name=model_name, model_nargs=model_nargs, story_path=story_path,
                                           rag_folder=rag_dir, verbose=verbose)
        except SystemExit:
            pass

    def send_prompt(self):
        if self.llm_model is None:
            self.chat_widget.append("Please create an LLM model!")
        else:
            # sys.stdout = EmittingStream()
            # sys.stdout.textWritten.connect(self.communication)
            self.chat_widget.append("User: ")
            self.chat_widget.append(self.prompt_send.text())
            self.chat_widget.append("----------------------------------------------")
            result = self.llm_model.invoke(self.prompt_send.text())
            self.chat_widget.append("Model: ")
            self.chat_widget.append(result)

    def model_list_clicked(self):
        model_text = self.available_models[self.listWidget.currentRow()]
        self.model_text_edit.setText(model_text)

    def communication(self, text):
        self.chat_widget.append(text)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MyWindow()
    widget.resize(1000, 800)
    widget.show()

    sys.exit(app.exec())
