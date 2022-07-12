# lang_split_methods

Dataset: 3000 texts, davon 1000 aus jeder source (Blogs, Twitter, Reddit)

methoden zum Splitten:

- nltk.sent_tokenize
- spacy nlp tokenization
- re_long: pattern = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
- re_short: pattern = "[.!?]"
    - für beide regex re.split methoden habe ich in der funktion noch sätze < 4 wörter entfernt, das ist in der runtime für die regex methoden mit drin
- stanza nlp tokenization

Methoden für language detection:

- langdetect
- textblob
    - hat komplett versagt, die funktion läuft hier nichtmehr, man soll die google api verwenden. habe eh nicht damit gerechnet dass das schnell genug ist,
    weil textblob übers web ne api anfragt, langsam ...
    - einige andere libraries sind nervig vong dependencies her ...
- langid


Resultate als graphen siehe graphs ordner!
