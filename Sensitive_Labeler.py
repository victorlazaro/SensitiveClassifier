import matplotlib.pyplot as plt
from Tkinter import *
from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from

class SensitiveLabeler(Labeler):

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)

    @inherit_docstring_from(Labeler)
    def label(self, text):
        print(text[0])
        label = raw_input("Label (0 or 1):")
        return self.label_name.index(label[0])






        # def onclick(event):
        #     return self.label_name.index(entry.get())
        #
        # top = Tk()
        # paragraphOutput = Text(top)
        # paragraphOutput.insert(INSERT, text[0])
        # paragraphOutput.config(state=DISABLED)
        # paragraphOutput.pack()
        # UserLabel = Label(top, text="Label (0 or 1):")
        # UserLabel.pack(side=LEFT)
        # entry = Entry(top, bd=5)
        # entry.pack(side=RIGHT)
        # top.mainloop()
        # top.bind('<Return>', onclick)
        # return self.label_name.index(entry.get())
        # if (len(entry.get()) > 0):
        #     return self.label_name.index(entry.get())




