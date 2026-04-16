import nltk

packages = [
    "punkt",
    "stopwords"
]

for pkg in packages:
    nltk.download(pkg)