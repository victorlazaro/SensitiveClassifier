# Sensitive Active Learning

This runs a demo of using active learning to try to classify sensitive vs
non-sensitive documents more quickly than random labeling.

## To Run

Note: These instructions expect this to be run on Linux.

If desired, create a virtual environment by typing 'virtualenv ENV' (or 
'virtualenv -p python2 ENV' if 'python' runs python 3 rather than python 2) in
the project folder. This creates a local copy of Python that will avoid
globally installing dependencies of this project.

You should then activate this environment by typing '. ENV/bin/activate'.

Then, type 'pip install -r requirements' to get the dependencies for the
project. If you run into problems installing the requirements, one thing to
try is running 'pip install --upgrade pip' to get the latest version of pip.

Finally, type 'python sensitive\_active\_learning' to start the demo.
