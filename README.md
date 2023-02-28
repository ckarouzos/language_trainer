# language_trainer
Utils for NLP, a training playground for language models.


## Requirements
poetry update
poetry export --without-hashes --format=requirements.txt > requirements.txt
sed -i 's/;.*//' requirements.txt

python3.8 -m pip install -r requirements.txt