from appJar import gui

class App:

	label = -1	

	def __init__(self, text):
		self.text = text

		app = gui()
	
		app.setFont(12)

		app.addMessage("Text to be labeled", text)

		app.setMessageSticky("Text to be labeled", "both")

		row = app.getRow()

		app.addButton("Sensitive", press)
		app.addButton("Not sensitive", press)

		app.go()

	def press(btn):
                if btn == "Sensitive":
                        label = 1
                else:
                        label = 0

