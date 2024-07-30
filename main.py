
import os
import re
import numpy as np
import cv2
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import PyPDF2

# Install necessary packages if running for the first time

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def extract_features(text):
    invoice_number = re.search(r"Rechnung\s*Nr\.?\s*[:\-]?\s*(\S+)", text, re.IGNORECASE)
    date = re.search(r"Datum\s*[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text, re.IGNORECASE)
    amount = re.search(r"Total\s*[:\-]?\s*\€?\$?(\d+[\.,]?\d{0,2})", text, re.IGNORECASE)

    features = {
        "invoice_number": invoice_number.group(1) if invoice_number else None,
        "date": date.group(1) if date else None,
        "amount": amount.group(1) if amount else None,
        "keywords": set(re.findall(r'\b\w+\b', text.lower()))
    }

    print(f"Extracted features: {features}")  # Debugging line to check extracted features
    return features

def calculate_cosine_similarity(features1, features2):
    vectorizer = TfidfVectorizer()
    keywords1 = ' '.join(features1['keywords'])
    keywords2 = ' '.join(features2['keywords'])
    tfidf_matrix = vectorizer.fit_transform([keywords1, keywords2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def calculate_jaccard_similarity(features1, features2):
    keywords1 = list(features1['keywords'])
    keywords2 = list(features2['keywords'])
    if not keywords1 or not keywords2:
        return 0
    vectorizer = CountVectorizer().fit_transform([' '.join(keywords1), ' '.join(keywords2)])
    vectors = vectorizer.toarray()
    jaccard_sim = jaccard_score(vectors[0], vectors[1], average='binary')
    return jaccard_sim

def extract_image_from_pdf(pdf_path):
    images = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj]._data
                        if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                            mode = "RGB"
                        else:
                            mode = "P"
                        img = Image.frombytes(mode, size, data)
                        images.append(img)
    return images

def save_image_from_pdf(pdf_path, output_path):
    images = extract_image_from_pdf(pdf_path)
    if images:
        images[0].save(output_path)

def calculate_image_similarity(image1_path, image2_path):
    image1 = cv2.imread(image1_path, 0)
    image2 = cv2.imread(image2_path, 0)

    if image1 is None or image2 is None:
        return 0

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 0:
        img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)
        plt.imshow(img3), plt.show()

    return len(matches)

database = []

def add_to_database(invoice_features, file_name, image_path):
    database.append((invoice_features, file_name, image_path))

def find_most_similar_invoice(new_invoice_features, new_image_path):
    highest_similarity = 0
    most_similar_invoice = None
    most_similar_file_name = None

    for stored_invoice, file_name, image_path in database:
        cosine_sim = calculate_cosine_similarity(new_invoice_features, stored_invoice)
        jaccard_sim = calculate_jaccard_similarity(new_invoice_features, stored_invoice)
        image_sim = calculate_image_similarity(new_image_path, image_path)

        combined_similarity = (cosine_sim + jaccard_sim + image_sim) / 3

        if combined_similarity > highest_similarity:
            highest_similarity = combined_similarity
            most_similar_invoice = stored_invoice
            most_similar_file_name = file_name
    return most_similar_invoice, highest_similarity, most_similar_file_name

def load_files_from_directory(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def process_files(file_paths):
    for file_path in file_paths:
        text = extract_text_from_pdf(file_path)
        features = extract_features(text)
        image_path = file_path.replace('.pdf', '.jpg')
        save_image_from_pdf(file_path, image_path)
        add_to_database(features, file_path, image_path)

# Define paths
training_directory = '/Users/swayansubaral/Documents/doc/train'
testing_directory = '/Users/swayansubaral/Documents/doc/test'

# Load training files
training_files = load_files_from_directory(training_directory)

# Train the system with the training set
process_files(training_files)

# Load testing files
testing_files = load_files_from_directory(testing_directory)

# Test the system with the test set and measure accuracy
correct_matches = 0
total_tests = len(testing_files)

for test_file in testing_files:
    print(f"\nTesting with file: {test_file}")
    test_text = extract_text_from_pdf(test_file)
    print(f"Extracted text: {test_text[:1000]}")  # Debugging line to check extracted text (first 1000 characters)
    test_features = extract_features(test_text)

    # Print extracted features for debugging
    print(f"Extracted Test Invoice Features: {test_features}")

    test_image_path = test_file.replace('.pdf', '.jpg')
    save_image_from_pdf(test_file, test_image_path)

    most_similar_invoice, similarity_score, most_similar_file_name = find_most_similar_invoice(test_features, test_image_path)

    if most_similar_invoice:
        print(f"Most similar invoice found in file: {most_similar_file_name} with similarity score: {similarity_score}")
        print("Test Invoice Details:")
        print(f"Invoice Number: {test_features['invoice_number']}")
        print(f"Date: {test_features['date']}")
        print(f"Amount: {test_features['amount']}")

        print("Most Similar Invoice Details:")
        print(f"Invoice Number: {most_similar_invoice['invoice_number']}")
        print(f"Date: {most_similar_invoice['date']}")
        print(f"Amount: {most_similar_invoice['amount']}")

        # Assume match is correct if similarity score is above a threshold
        if similarity_score > 0.5:  # Adjust threshold based on your needs
            correct_matches += 1
    else:
        print("No similar invoice found.")

accuracy = correct_matches / total_tests
print(f"\nAccuracy: {accuracy * 100:.2f}%")