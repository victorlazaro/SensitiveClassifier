from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from
from appJar import gui

class SensitiveLabeler(Labeler):

    lbl = None
    app = None

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)
        self.paragraphs = kwargs.pop('paragraphs', None)



    def press(self, btn):
        if btn == "Sensitive":
            self.lbl = "1"
        elif btn == "Not sensitive":
            self.lbl = "0"
        else:
            self.lbl = "-1"
        self.app.stop()
        self.app = None

    @inherit_docstring_from(Labeler)
    def label(self, text, index):

        self.app = gui()

        self.app.setFont(12)

        self.app.addMessage("Text to be labeled", self.paragraphs[index])

        self.app.setMessageSticky("Text to be labeled", "both")

        self.app.addButton("Sensitive", self.press)
        self.app.addButton("Not sensitive", self.press)

        self.app.go()
        if self.lbl is None:
            return -1
        return self.label_name.index(self.lbl)







