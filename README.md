# Text2KG

UWA's graph visualisation app that was submitted as part of the ICDM 2019 ICDM/ICBK Knowledge Graph Contest.

Team members:

- Wei Liu
- Majigsuren Enkhsaikhan
- Michael Stewart
- Morgan Lewis
- Thomas Smoker

## File structure
    
    sourcecode
        visualisation           // Visualisation web app
        requirements.txt        // All Python requirements for the /sourcecode directory
        setup.sh                // Creates a virtual environment and installs required packages
       

## Running the code

This repository contains the sourcecode for the visualisation app, but our triple generation code is not publically available. In order to run the visualisation app, you must first place a triple generation model into `sourcecode/candidate extraction/triples_from_text.py`. This file requires a function named `extract_triples`, which returns a list of triples given a document. For example, if given the document `"Barrack Obama went to the Whitehouse with Michelle"`, the function must take that string as input and return a list of triples, e.g. `[["Barrack", "went to", "the Whitehouse"], ["Barrack Obama", "with", "Michelle"]... ]`.

Once the triple generation model is in place, you may install all required packages via:

    cd sourcecode
    chmod u+x setup.sh
    ./setup.sh

Please ensure you are using Python 3.6 and have the `virtualenv` package installed in order to maximise compatibility. Running the shell scripts will likely require a Unix-based OS such as Ubuntu.

## Visualisation web app

Our Flask application that provides an easy-to-use interface for knowledge graph construction from text. Users may enter documents into a text box, click on the "Create graph" button, and their documents will be sent to our Candidate Extraction model for processing. An interactive graph will then be quickly be drawn to the screen.

If you wish to run the server locally, you must run the Visualisation server via:

    $ cd sourcecode/visualisation
    $ chmod u+x run_server.sh
    $ ./run_server.sh

The visualisation app will then be available at `localhost:5000` in a web browser.

You can also run `./run_server_gunicorn.sh` if you wish to run the server via wsgi.


## License

text2kg-uwa (c) by Michael Stewart, Majigsuren Enkhsaikhan and Wei Liu

text2kg-uwa is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
