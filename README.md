# AI-Negotiation

Uses python 3.6

## Data
Source of the dataset from Facebook AI Research: https://code.facebook.com/posts/1686672014972296/deal-or-no-deal-training-ai-bots-to-negotiate/



## Virtual Environment
To create virtual environment:

>```
>sudo pip install virtualenv        # If virtualenv has not already been installed
>virtualenv -p python3 .env         # To create a python3 virtual environment
>source .env/bin/activate           # Activate the virtual environment
>pip install -r requirements.txt    # Install all dependencies (only need to run once)
>......
>deactivate                         # Exit virtual environment
>```

## Files

Use `readData.py` to process the data from .txt files to a highly-formatted list of dictionaries for each example. 
The formatted data is saved in .json files

All of the models are defined in `rnn.py`

Train the negotiation output classifier using `trainOutputClassification.py`

Train the encoder and decoder RNNs to generate natural langauge responses in negotiation using `trainGenerative.py`

Converse with negotiation bots, either one operating by handcrafted rules or the RNN encoder/decoder bot using `converse.py`
