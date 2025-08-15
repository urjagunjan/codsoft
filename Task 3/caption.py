import os
import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sqlite3
from typing import List, Dict, Any, Optional, Set
import random
import json

# Keras imports - Placed here to avoid issues with GPU configuration
import tensorflow as tf

# Keras imports
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, custom_object_scope
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer

# Database setup
class CaptionDatabase:
    def __init__(self, db_path: str = 'captions.db'):
        self.conn = sqlite3.connect(db_path, timeout=20.0, detect_types=sqlite3.PARSE_DECLTYPES)
        self._init_db()
        
    def _init_db(self):
        cursor = self.conn.cursor()
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            features BLOB
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            caption TEXT,
            is_training BOOLEAN,
            FOREIGN KEY (image_id) REFERENCES images (image_id)
        )
        ''')
        self.conn.commit()
    
    def save_features(self, image_id: str, features: np.ndarray):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO images (image_id, features)
        VALUES (?, ?)
        ''', (image_id, features.tobytes()))
        self.conn.commit()
    
    def get_features(self, image_id: str) -> Optional[np.ndarray]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT features FROM images WHERE image_id = ?', (image_id,))
        result = cursor.fetchone()
        if result:
            return np.frombuffer(result[0], dtype=np.float32).reshape(1, -1)
        return None
    
    def save_caption(self, image_id: str, caption: str, is_training: bool = True):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO captions (image_id, caption, is_training)
        VALUES (?, ?, ?)
        ''', (image_id, caption, is_training))
        self.conn.commit()
    
    def get_captions(self, image_id: str = None, is_training: bool = None) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        query = 'SELECT image_id, caption, is_training FROM captions'
        params = []
        
        conditions = []
        if image_id is not None:
            conditions.append('image_id = ?')
            params.append(image_id)
        if is_training is not None:
            conditions.append('is_training = ?')
            params.append(is_training)
            
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
            
        cursor.execute(query, params)
        return [{'image_id': row[0], 'caption': row[1], 'is_training': bool(row[2])} for row in cursor.fetchall()]
    
    def get_image_ids(self, is_training: bool = None) -> List[str]:
        cursor = self.conn.cursor()
        if is_training is not None:
            cursor.execute('SELECT DISTINCT image_id FROM captions WHERE is_training = ?', (is_training,))
        else:
            cursor.execute('SELECT DISTINCT image_id FROM captions')
        return [row[0] for row in cursor.fetchall()]
    
    def close(self):
        self.conn.close()

# Initialize database connection
db = CaptionDatabase()

# --- Constants ---
# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(BASE_DIR, 'working')
MODEL_WEIGHTS_FILE = os.path.join(WORKING_DIR, 'model_weights.h5')

# Create working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# --- 1. Feature Extraction ---
def extract_features(directory: str, batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Extract features from all images in the directory."""
    model = VGG16(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    
    # Get list of image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting features"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_ids = []
        
        # Load and preprocess batch
        for img_name in batch_files:
            try:
                img_path = os.path.join(directory, img_name)
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                batch_images.append(image[0])
                batch_ids.append(os.path.splitext(img_name)[0])
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        
        if not batch_images:
            continue
            
        # Process batch
        batch_images = np.array(batch_images)
        batch_features = model.predict(batch_images, verbose=0)
        
        # Store features in a dictionary
        for img_id, feature in zip(batch_ids, batch_features):
            features[img_id] = feature
            
    print(f"Processed {len(image_files)} images")
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

def load_clean_descriptions_to_db(descriptions: Dict[str, List[str]], dataset: Set[str], is_training: bool):
    """Loads cleaned descriptions into the database in batches."""
    cursor = db.conn.cursor()
    captions_to_insert = []
    for image_id, desc_list in descriptions.items():
        if image_id in dataset:
            for desc in desc_list:
                captions_to_insert.append((image_id, desc, is_training))
    
    if captions_to_insert:
        cursor.executemany('''
        INSERT INTO captions (image_id, caption, is_training)
        VALUES (?, ?, ?)
        ''', captions_to_insert)
        db.conn.commit()

# --- 3. Model Training ---
def to_lines() -> List[str]:
    """Get all captions from the database."""
    captions = db.get_captions()
    return [c['caption'] for c in captions]

def create_tokenizer():
    """Create a tokenizer from all captions in the database."""
    lines = to_lines()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_len() -> int:
    """Get the maximum length of captions in the database."""
    captions = to_lines()
    return max(len(d.split()) for d in captions) if captions else 0

def create_caption_sequences(tokenizer, captions):
    """Pre-processes captions into a list of (image_id, sequence) pairs."""
    sequences = []
    for cap_data in captions:
        seq = tokenizer.texts_to_sequences([cap_data['caption']])[0]
        sequences.append((cap_data['image_id'], seq))
    return sequences

def data_generator(sequences, features, tokenizer, max_length, batch_size, vocab_size):
    """Data generator for model training."""
    while True:
        random.shuffle(sequences)
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            X1, X2, y = [], [], []
            for img_id, seq in batch_seqs:
                photo = features.get(img_id)
                if photo is None:
                    continue
                for j in range(1, len(seq)):
                    in_seq, out_seq = seq[:j], seq[j]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo.reshape(4096))
                    X2.append(in_seq)
                    y.append(out_seq)
            yield (np.array(X1), np.array(X2)), np.array(y)

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

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
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
    print("Starting Image Captioning AI...")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # --- File Paths ---
        IMAGE_DIR = os.path.join(BASE_DIR, 'Flicker8k_Dataset')
        TOKENIZER_FILE = os.path.join(WORKING_DIR, 'tokenizer.pkl')
        CAPTIONS_FILE = os.path.join(BASE_DIR, 'Flickr8k.token.txt')
        TRAIN_FILE = os.path.join(BASE_DIR, 'Flickr_8k.trainImages.txt')
        TEST_FILE = os.path.join(BASE_DIR, 'Flickr_8k.testImages.txt')
        DEV_FILE = os.path.join(BASE_DIR, 'Flickr_8k.devImages.txt')
        FEATURES_FILE = os.path.join(WORKING_DIR, 'features.pkl')

        print("\nChecking required files and directories...")
        for path in [IMAGE_DIR, WORKING_DIR]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")
            else:
                print(f"Directory exists: {path}")

        # --- Step 1: Feature Extraction ---
        if os.path.exists(FEATURES_FILE):
            print(f"Loading features from {FEATURES_FILE}...")
            with open(FEATURES_FILE, 'rb') as f:
                features = pickle.load(f)
            print(f"Loaded {len(features)} features.")
        else:
            print("Extracting features... This will take a while.")
            features = extract_features(IMAGE_DIR)
            print(f"Extracted {len(features)} features. Saving to {FEATURES_FILE}...")
            with open(FEATURES_FILE, 'wb') as f:
                pickle.dump(features, f)

        # --- Step 2: Prepare training data ---
        doc = load_doc(CAPTIONS_FILE)
        descriptions = load_descriptions(doc)
        clean_descriptions(descriptions)
        train_img_ids = list(load_set(TRAIN_FILE))
        test_img_ids = list(load_set(TEST_FILE))

        print(f'Training images: {len(train_img_ids)}')
        print(f'Test images: {len(test_img_ids)}')

        # Clear and load captions into database
        print("Clearing old captions from the database...")
        db.conn.execute('DELETE FROM captions')
        db.conn.commit()

        print('Loading training captions into DB...')
        load_clean_descriptions_to_db(descriptions, set(train_img_ids), is_training=True)
        print('Loading test captions into DB...')
        load_clean_descriptions_to_db(descriptions, set(test_img_ids), is_training=False)

        # --- Step 3: Tokenizer and Model Setup ---
        tokenizer = create_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        max_length = max_len()
        print(f'Vocabulary size: {vocab_size}')
        print(f'Max caption length: {max_length}')

        with open(TOKENIZER_FILE, 'wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Creating new model...")
        model = define_model(vocab_size, max_length)

        epochs = 1
        batch_size = 32
        train_captions = db.get_captions(is_training=True)
        steps_per_epoch = len(train_captions) // batch_size

        if steps_per_epoch == 0:
            print("Error: Not enough training data.")
            db.close()
            exit()

        print("Preparing training data sequences...")
        train_captions = db.get_captions(is_training=True)
        train_sequences = create_caption_sequences(tokenizer, train_captions)
        print(f"Created {len(train_sequences)} sequences.")

        steps_per_epoch = len(train_sequences) // batch_size
        if steps_per_epoch == 0:
            print("Error: Not enough training data to form a single batch.")
            db.close()
            exit()

        MODEL_FILE = 'model.keras'
        if os.path.exists(MODEL_FILE):
            print(f"Loading existing model from {MODEL_FILE}...")
            model = load_model(MODEL_FILE)
        else:
            print("Training new model...")
            train_gen = data_generator(train_sequences, features, tokenizer, max_length, batch_size, vocab_size)
            model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)
            model.save(MODEL_FILE)
            print(f"Model training complete and saved to {MODEL_FILE}")

        # --- Step 4: Generate Caption for a Test Image ---
        if not test_img_ids:
            print("No test images found.")
        else:
            # Filter test_img_ids to only include images that exist on disk
            existing_test_images = [img_id for img_id in test_img_ids if os.path.exists(os.path.join(IMAGE_DIR, f"{img_id}.jpg"))]
            if not existing_test_images:
                print("No valid test images found on disk.")
                random_image_id = None
            else:
                random_image_id = random.choice(existing_test_images)

        if random_image_id:
            print(f"\nGenerating caption for image: {random_image_id}")
            image_path = os.path.join(IMAGE_DIR, f"{random_image_id}.jpg")
            
            if os.path.exists(image_path):
                image = load_img(image_path, target_size=(224, 224))
                image.save('last_test_image.jpg')
                print(f"Test image saved as 'last_test_image.jpg'")

                feature = features.get(random_image_id)
                if feature is not None:
                    description = generate_desc(model, tokenizer, feature, max_length)
                    print(f'\nGenerated Caption: {description}')
                    
                    actual_captions = [c['caption'].replace('startseq', '').replace('endseq', '').strip() for c in db.get_captions(image_id=random_image_id)]
                    print('\nActual captions:')
                    for i, cap in enumerate(actual_captions, 1):
                        print(f"{i}. {cap}")
                else:
                    print(f"Error: Could not find features for image {random_image_id}.")
            else:
                print(f"Error: Image file not found at {image_path}")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        db.close()
