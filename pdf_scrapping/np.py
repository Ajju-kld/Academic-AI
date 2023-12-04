import fitz  # PyMuPDF
from transformers import pipeline

def extract_titles_and_paragraphs(pdf_path):
    titles = []
    paragraphs = []

    with open(pdf_path, 'rb') as file:
        pdf_text = extract_text_with_pymupdf(file)

    lines = pdf_text.split('\n')
    current_title = ""
    current_paragraph = ""

    for line in lines:
        if line.strip():
            # Assume lines starting with a number or bullet point are titles
            if line[0].isdigit() or (len(line) > 1 and line[1] == '.'):
                if current_title and current_paragraph:
                    titles.append(current_title.strip())
                    paragraphs.append(current_paragraph.strip())
                current_title = line
                current_paragraph = ""
            else:
                current_paragraph += line + '\n'

    # Add the last title and paragraph
    if current_title and current_paragraph:
        titles.append(current_title.strip())
        paragraphs.append(current_paragraph.strip())

    return titles, paragraphs, pdf_text

def main():
    # Replace with the path to your PDF file
    pdf_path = "pdf/ktu.pdf"

    # Extract titles, paragraphs, and full text from the PDF
    titles, paragraphs, pdf_text = extract_titles_and_paragraphs(pdf_path)

    print("Titles:")
    print(titles)

    print("\nParagraphs:")
    print(paragraphs)

    # Load pre-trained model for question answering
    question_answering = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

    # Ask a question about a specific title or paragraph
    question = input("Ask a question about a title or paragraph: ")

    # Get the answer
    answer = question_answering(question=question, context=pdf_text)

    print("Answer:", answer['answer'])

if __name__ == "__main__":
    main()
