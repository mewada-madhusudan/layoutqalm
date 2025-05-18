from transformers import pipeline
import torch
import gradio as gr
import os
from PIL import Image
import pytesseract
import tempfile
import shutil
from pdf2image import convert_from_path

model_name = "deepset/roberta-base-squad2"
text_qna = pipeline("question-answering", model=model_name, tokenizer=model_name)
vision_qna = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

# Vision QnA requires: PyTesseract for OCR. Tesseract executable needs to be installed separately.
# sudo apt install tesseract-ocr (https://tesseract-ocr.github.io/tessdoc/Installation.html)

def load_file(file_input, encoding = 'utf-8'):  
  if not os.path.exists(file_input):
    raise FileNotFoundError(f"The file does not exist.")

  with open(file_input, 'r', encoding=encoding) as file:
    try:
      content = file.read()
    except UnicodeDecodeError:
      # If a UnicodeDecodeError occurs, try reading with 'latin1' encoding
      with open(file_input, 'r', encoding='latin1') as file:
        content = file.read()

  return content

def save_image(file):
    try:
      temp_dir = tempfile.mkdtemp()
      file_path = os.path.join(temp_dir, os.path.basename(file.name))
      # Copy the file from the temporary Gradio directory to our temporary directory
      shutil.copyfile(file.name, file_path)
      # when working with saving image files through Gradio, 
      # using `shutil.copyfile` to handle `NamedString` objects for file uploads is the correct approach

      return file_path
    except Exception as e:
      print(e)


def save_pdf(file):
  temp_dir = tempfile.mkdtemp()
  pdf_path = os.path.join(temp_dir, os.path.basename(file.name))
  
  # Copy the file from the temporary Gradio directory to our temporary directory
  shutil.copyfile(file.name, pdf_path)
  
  # Convert PDF to images
  images = convert_from_path(pdf_path)
  
  image_paths = []
  for i, img in enumerate(images):
      image_path = os.path.join(temp_dir, f'page_{i}.png')
      img.save(image_path, 'PNG')
      image_paths.append(image_path)
  
  print(image_paths)
  return image_paths


def qna_text_content(content, question):
  result = text_qna(question=question, context=content)
  return result

def qna_image_content(content, question):
  # result = vision_qna(question=question, image=content)
  result = vision_qna(content, question)
  print(f"image question: {question}")
  return result

def qna_pdf_content(image_paths, question):
  answers = []
  try:
      for image_path in image_paths:
          result = vision_qna(image=image_path, question=question)
          print(result[0]['answer'], result[0]['score'])
          answers.append(result[0]['answer'])
      return " \n".join(answers)
  except Exception as e:
      return f"An error occurred during processing: {e}"

def answer_the_question_for_doc(text_input, file_input, question):
  #  Order of input parameters is Imp. for Gradio to accept respective Inputs 
  if file_input is not None:
    print(f"File type: {type(file_input)}")
    print(f"File name: {file_input.name}")

    file_extension = file_input.name.split('.')[-1].lower()
    if file_extension in ['txt']:
      try:
        content = load_file(file_input)
        if not content or not question:
            return "Please provide both content and a question."
        result = qna_text_content(content, question)
        return result["answer"]
      
      except FileNotFoundError or Exception as e:
        print(e)
        exit(1)
    
    elif file_extension in ['png', 'jpeg', 'jpg']:
      try:
        img_file_path = save_image(file_input)
        
        if not question:
            return "Please provide a question."
        result = qna_image_content(img_file_path, question)
        print(result)
        return result[0]["answer"]

      except Exception as e:
        return f"An error occurred during vision processing: {e}" 

    elif file_extension in ['pdf']:
      try:
        image_paths = save_pdf(file_input)
        
        if not question:
            return "Please provide a question."
        result = qna_pdf_content(image_paths, question)
        print(result)
        return result

      except Exception as e:
        return f"An error occurred during vision processing: {e}" 

    else:
      return "Unsupported file type. Please upload a .txt, ,.pdf, .png, or .jpeg file."
  else:
    if not text_input or not question:
      return "Please provide both content and a question."
    content = text_input
    result = qna_text_content(content, question)
    return result["answer"]

gr.close_all()

with gr.Blocks() as demo:
    gr.Markdown("# QnA System") 
    gr.Markdown("This App answers a question based on text content or uploaded file (txt, png, jpeg, pdf).")

    with gr.Row():
        text_input = gr.Textbox(label="Text Input", placeholder="Enter text content here...")
        file_input = gr.File(label="File Upload", file_types=['txt', 'png', 'jpeg', 'pdf'])

    question = gr.Textbox(label="Question", placeholder="Enter your question here...")
    output = gr.Textbox(label="Answer", placeholder="The answer will appear here...")

    text_input.change(lambda x: gr.update(visible=not x), inputs=text_input, outputs=file_input)
    file_input.change(lambda x: gr.update(visible=not x), inputs=file_input, outputs=text_input)

    button = gr.Button("Get Answer")
    button.click(answer_the_question_for_doc, inputs=[text_input, file_input, question], 
                outputs=output)
demo.launch()

# print(qna_image_content("https://gradientflow.com/wp-content/uploads/2023/10/newsletter87-RAG-simple.png", "What is the step prior to embedding?"))