import os
import pickle
import string
import os
import random
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# --- Constants ---
BASE_DIR = 'C:\\Users\\urjag\\CascadeProjects\\ImageCaptioningAI'
WORKING_DIR = os.path.join(BASE_DIR, 'working')
MODEL_WEIGHTS_FILE = os.path.join(WORKING_DIR, 'model_weights.h5')

# Create working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# --- 1. Feature Extraction ---
def extract_features(directory):
    model = VGG16(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
    return features

# --- 2. Load and Prepare Text Data ---
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_descriptions(doc):
    mapping = {}
    for line in doc.strip().split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    return mapping

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = 'startseq ' + ' '.join(desc) + ' endseq'

def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.strip().split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = {}
    for line in doc.strip().split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

# --- 3. Model Training ---
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_len(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def data_generator(descriptions, photos, tokenizer, max_len, vocab_size):
    while 1:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield (photo, in_seq), out_seq

def create_sequences(tokenizer, max_len, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_len):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

# --- 4. Caption Generation (Inference) ---
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_len):
    in_text = 'startseq'
    for i in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# --- Main Execution ---
if __name__ == '__main__':
    # --- File Paths ---
    IMAGE_DIR = os.path.join(BASE_DIR, 'Flicker8k_Dataset')
    CAPTIONS_FILE = os.path.join(BASE_DIR, 'Flickr8k.token.txt')
    TRAIN_FILE = os.path.join(BASE_DIR, 'Flickr_8k.trainImages.txt')
    FEATURES_FILE = os.path.join(WORKING_DIR, 'features.pkl')
    TOKENIZER_FILE = os.path.join(WORKING_DIR, 'tokenizer.pkl')

    # --- Step 1: Feature Extraction ---
    if not os.path.exists(FEATURES_FILE):
        if not os.path.exists(IMAGE_DIR):
            print(f"Error: Image directory not found at {IMAGE_DIR}")
            exit()
        print("Extracting features... This will take a while.")
        features = extract_features(IMAGE_DIR)
        print('Extracted Features:', len(features))
        with open(FEATURES_FILE, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(FEATURES_FILE, 'rb') as f:
            features = pickle.load(f)
        print('Loaded features:', len(features))

    # --- Step 2: Prepare training data ---
    if not os.path.exists(CAPTIONS_FILE) or not os.path.exists(TRAIN_FILE):
        print(f"Error: Required dataset files (captions or train split) not found.")
        exit()

    doc = load_doc(CAPTIONS_FILE)
    descriptions = load_descriptions(doc)
    clean_descriptions(descriptions)
    train = list(load_set(TRAIN_FILE))
    train = train[:1000]
    print(f'Descriptions: train={len(train)} (using subset of 1000 images)')
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print(f'Photos with descriptions: {len(train_descriptions)}')

    tokenizer = create_tokenizer(train_descriptions)
    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size:', vocab_size)

    max_len = max_len(train_descriptions)
    print('Description Length:', max_len)

    # --- Step 3: Train or Load Model ---
    model = define_model(vocab_size, max_len)
    if os.path.exists('model_final.h5'):
        print("Loading existing model weights...")
        model.load_weights('model_final.h5')
    else:
        print("Training new model...")
        model = define_model(vocab_size, max_len)
        epochs = 10

        # Create a tf.data.Dataset
        def generator_wrapper():
            return data_generator(train_descriptions, train_features, tokenizer, max_len, vocab_size)

        dataset = tf.data.Dataset.from_generator(
            generator_wrapper,
            output_signature=(
                (tf.TensorSpec(shape=(4096,), dtype=tf.float32), tf.TensorSpec(shape=(max_len,), dtype=tf.int32)),
                tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
            )
        ).batch(64).prefetch(tf.data.AUTOTUNE)

        num_sequences = sum(len(desc.split()) - 1 for descs in train_descriptions.values() for desc in descs)
        steps_per_epoch = num_sequences // 64

        model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
        model.save('model_final.h5')

    # --- Step 4: Generate a caption for a random test image ---
    print('\n--- Generating Caption for a Random Test Image ---')

    # Load test image names
    test_img_keys = load_set('Flickr_8k.testImages.txt')
    random_image_id = random.choice(list(test_img_keys))
    print(f"Image ID: {random_image_id}")

    # Load the image and save it for viewing
    image_path = os.path.join(IMAGE_DIR, random_image_id + '.jpg')
    image = load_img(image_path, target_size=(224, 224))
    image.save('last_test_image.jpg')
    print(f"Image saved as 'last_test_image.jpg'")

    # Extract its features
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    image_array = img_to_array(image)
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
    image_array = preprocess_input(image_array)
    feature = vgg_model.predict(image_array, verbose=0)

    # Generate caption
    description = generate_desc(model, tokenizer, feature, max_len)
    print(f'Generated Caption: {description}')
