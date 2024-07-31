#Overview

This project aims to develop a system that identifies and matches similar documents, specifically invoices, based on their content and structure. The system extracts text from PDF invoices, calculates similarities using various metrics (Cosine Similarity, Jaccard Similarity, and Image Similarity), and matches incoming invoices against a database of existing invoices.

#Approach

##Document Representation Method

Text Extraction:
Text content is extracted from PDF files using the PyPDF2 library.
Extracted text is then used to identify relevant features such as invoice number, date, amount, and keywords.
Feature Extraction:
The following features are extracted from the text:
Invoice Number: Extracted using regex patterns.
Date: Extracted using regex patterns.
Amount: Extracted using regex patterns.
Keywords: Extracted by tokenizing the text and converting it to lowercase.
Image Extraction:
Images from PDF files are extracted using PyPDF2 and saved as image files.

Similarity Metrics Used

Cosine Similarity:
Utilizes TF-IDF vectors of keywords to calculate the cosine similarity between two invoices.
Jaccard Similarity:
Measures the overlap between the sets of keywords from two invoices.
Image Similarity:
Uses ORB (Oriented FAST and Rotated BRIEF) feature matching in OpenCV to compare the visual layout of invoices.

How to Run the Code

Prerequisites

Ensure the following Python libraries are installed:

PIP INSTALL -r req.txt

Project Structure
Ensure your project directory has the following structure:

scss

project_directory/
│
├── train/
│   ├── invoice1.pdf
│   ├── invoice2.pdf
│   └── ... (other training invoices)
│
├── test/
│   ├── invoice_test1.pdf
│   ├── invoice_test2.pdf
│   └── ... (other testing invoices)
│
├── main.py
└── README.md

Running the Code

1.Navigate to Project Directory:

Open a terminal and navigate to the project directory:

bash

cd /path/to/your/project_directory

2.Run the Script:

Execute the main script:

bash

python main.py



