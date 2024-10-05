import spacy
import unicodedata
from spacy.tokens import DocBin
from spacy.training import Example

# Load BIO Data
def load_bio_data(file_path):
    sentences = []
    sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, label = line.split()
                sentence.append((word, label))
    
    if sentence:  # Add the last sentence if the file doesn’t end with a blank line
        sentences.append(sentence)
    
    return sentences

def bio_to_spacy_format(bio_sentences):
    spacy_data = []
    
    for sentence in bio_sentences:
        text = " ".join([token for token, _ in sentence])
        entities = []
        start = 0
        
        for token, label in sentence:
            token_len = len(token)
            if label.startswith("B-"):
                entity_label = label[2:]
                entities.append((start, start + token_len, entity_label))
            start += token_len + 1  # +1 for the space
        
        spacy_data.append((text, {"entities": entities}))
    
    return spacy_data


# Load Character-Level Data
def load_char_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            text = lines[i].strip()
            if i + 2 < len(lines):
                entities = eval(lines[i + 1].strip())

                entity_spans = [(start, end, label) for start, end, label, _ in entities]
                data.append((text, {"entities": entity_spans}))
                # entities.sort(key=lambda x: x[0])
                # entity_spans = []
                # for _, _, label, entity in entities:
                #     start = None
                #     if entity_spans:
                #         start = entity_spans[-1][1]
                #     normalized_start = text.find(entity, start)
                #     if normalized_start == -1:
                #         print(entity, entity_spans)
                #     normalized_end = normalized_start + len(entity)
                    
                #     entity_spans.append((normalized_start, normalized_end, label))

                # data.append((text, {"entities": entity_spans}))
    
    return data

# Convert the loaded data to a DocBin
def create_training_data(spacy_data, nlp):
    db = DocBin()
    for text, annotations in spacy_data:
        doc = nlp.make_doc(text)
        ents = [doc.char_span(start, end, label=label) for start, end, label in annotations["entities"]]
        if None in ents:
            ents = []
        doc.ents = ents
        db.add(doc)
    return db

# Load and process both BIO and character-level data
train_bio_file = "everest/EverestNER-train-bio.txt"
train_char_file = "everest/EverestNER-train-char.txt"

train_bio_sentences = load_bio_data(train_bio_file)
train_spacy_bio_data = bio_to_spacy_format(train_bio_sentences)

train_spacy_char_data = load_char_data(train_char_file)

# Combine both the BIO and character-level data
train_spacy_data = train_spacy_bio_data + train_spacy_char_data

# Create blank model
nlp = spacy.blank("xx")  # Create a blank language model
ner = nlp.add_pipe('ner')

# Add labels to the NER pipeline
for _, annotations in train_spacy_data:
    for start, end, label in annotations["entities"]:
        ner.add_label(label)

# Convert the combined data to spaCy format and save
train_db = create_training_data(train_spacy_data, nlp)
train_db.to_disk("./train.spacy")  # Save the data for later use

# Fine-tune the model with both data formats
optimizer = nlp.begin_training()
for i in range(10):  # Number of epochs
    losses = {}
    # random.shuffle(train_spacy_data)
    batches = spacy.util.minibatch(train_spacy_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)  # Create Example object
            nlp.update([example], sgd=optimizer, losses=losses)  # Pass a list of Example objects
    print(f"Losses at iteration {i}: {losses}")

# Save the trained model
nlp.to_disk("satya")
