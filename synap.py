import pymupdf
import pandas as pd
import fitz
import numpy as np
import os
import json
import pickle
import shutil
from pyvis.network import Network
import networkx as nx
import copy
import logging
import google.generativeai as genai
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

genai.configure(api_key="AIzaSyCtjwJEDBUttvJ_dnjjCG8WHv98nVGYWLE")

model = genai.GenerativeModel('gemini-pro')


def download_images_per_page(doc, doc_name, page, page_index, DPI):
    image_list = page.get_images()
    if image_list:
        print(f"Found {len(image_list)} images on page {page_index}")
    else:
        print("No images found on page", page_index)

    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        pix = pymupdf.Pixmap(doc, xref)

        print('Image:')
        print(type(img))

        pix.save("./Parsed_PDF_Output/" + doc_name + "/" + "page_" + str(page_index + 1) + "/image_%s.jpg" % (
            image_index))
        pix = None

    return True


def extract_text_tables_images_per_page(doc, doc_name, doc_img, index, download, page_dir):
    print("IN extract_text_tables_images_per_page")
    page_dict = {}
    page_image_dict = {}
    tab_df_lst = []
    page = doc[index]
    tabs = page.find_tables()

    page_image_dict = extract_images_per_page(doc_img, doc_name, index, download)

    page_dict['page'] = page_image_dict['page']

    page_dict['img_cnt'] = page_image_dict['img_cnt']
    page_dict['img_npy_lst'] = page_image_dict['img_npy_lst']


    text = page.get_text()
    page_dict['text'] = text

    if tabs.tables == []:
        print('Do Nothing')
    else:
        for tab in tabs:
            df = pd.DataFrame()
            df = tab.to_pandas()
            tab_df_lst.append(df)

    page_dict['tables'] = tab_df_lst

    return page_dict


def extract_images_per_page(doc, doc_name, page_index, download):
    print("IN extract_images_per_page")
    page_image_dict = {}
    page_number = page_index + 1
    page = doc[page_index]

    image_list = page.get_images(full=True)

    img_cnt = len(image_list)
    npy_img_lst = []
    DPI = 150
    title = ""

    for image_index, img in enumerate(image_list, start=1):
        img_meta_dict = {}
        xref = img[0]
        smask = img[1]
        width = img[2]
        height = img[3]
        num_bits = img[4]
        colorspace = img[5]
        alt_colorspace = img[6]
        sym_name = img[7]
        img_filter = img[8]
        img_ref = img[9]

        img_meta_dict["img_obj_num"] = xref
        img_meta_dict["smask_obj_num"] = smask
        img_meta_dict["width"] = width
        img_meta_dict["height"] = height
        img_meta_dict["num_bits"] = num_bits
        img_meta_dict["colorspace"] = colorspace
        img_meta_dict["alt_colorspace"] = alt_colorspace
        img_meta_dict["sym_name"] = sym_name
        img_meta_dict["filter"] = img_filter
        img_meta_dict["referencer"] = img_ref

        pix = pymupdf.Pixmap(doc, xref)

        if pix.n - pix.alpha > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

        print("PIX BUFFER SIZE")
        print(len(pix.samples_mv))

        image_size = (pix.h * pix.w * 3)
        print('Original IMG_BUFFER_SIZE')
        print(image_size)

        print('Page Image Buffer Size')
        print(pix.samples_mv)

        try:
            img = np.ndarray([pix.h, pix.w, 3], dtype=np.uint8, buffer=pix.samples_mv)
            img_meta_dict["img_matrix"] = img
            img_meta_dict_copy = copy.deepcopy(img_meta_dict)  # Deep copy here
            npy_img_lst.append(img_meta_dict_copy)

        except Exception as e:
            logging.error(f"Image processing error on page {page_index + 1}, image {image_index + 1}: {e}")

            continue

        finally:
            pix = None
    page_image_dict['page'] = page_number
    page_image_dict['img_cnt'] = len(image_list)
    page_image_dict['img_npy_lst'] = npy_img_lst

    if download:
        download_images_per_page(doc, doc_name, page, page_index, DPI)

    return page_image_dict

def extract_text_images_tables(doc_path, download=False):
    doc_per_page_data_lst = []
    doc = fitz.open(doc_path)
    num_pages = len(doc)
    doc_img = pymupdf.open(doc_path)

    doc_name = os.path.basename(doc_path)
    doc_name_wo_type = os.path.splitext(doc_name)[0]
    doc_type = os.path.splitext(doc_name)[1][1:]

    output_dir = "Parsed_PDF_Output"
    doc_output_dir = os.path.join(output_dir, doc_name_wo_type)

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(doc_output_dir):
        shutil.rmtree(doc_output_dir)
    os.makedirs(doc_output_dir)

    for i in range(num_pages):
        page_dir = os.path.join(doc_output_dir, f"page_{i + 1}")
        os.makedirs(page_dir)

        page_data = extract_text_tables_images_per_page(doc, doc_name_wo_type, doc_img, i, download, page_dir)
        doc_per_page_data_lst.append(page_data)

    document_dictionary = {"name": doc_name, "type": doc_type, "data": doc_per_page_data_lst}

    if download:
        pickle_path = os.path.join(doc_output_dir, f"{doc_name}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(document_dictionary, f)
    doc.close()
    doc_img.close()
    return document_dictionary

def preprocess_text(text):
    text = " ".join(text.split())
    return text

def summarize_text(text, max_tokens=100000):
    try:
        response = model.generate_content(f"Summarize the following text in {max_tokens} tokens or less: {text}")
        return response.text
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

def extract_insights(text):
    try:
        response = model.generate_content(f"Extract key insights from the following text: {text}")
        return response.text
    except Exception as e:
        print(f"Error during insight extraction: {e}")
        return None

def combine_text_from_pdf(extracted_data):
    combined_text = ""
    for page_data in extracted_data['data']:
        text = page_data['text']
        combined_text += preprocess_text(text) + "\n"
    return combined_text


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return "No file part"
        file = request.files['pdf_file']
        if file.filename == '':
            return "No selected file"

        if file and file.filename.lower().endswith('.pdf'):  # Check if it's a PDF
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                extracted_data = extract_text_images_tables(filepath, download=False)  # Don't download images for web
                combined_text = combine_text_from_pdf(extracted_data)

                if not combined_text.strip():
                    return "Error: No text extracted from PDF."

                MAX_SECTION_LENGTH = 50000
                sections = [combined_text[i:i + MAX_SECTION_LENGTH] for i in range(0, len(combined_text), MAX_SECTION_LENGTH)]

                all_summaries = []
                all_insights = []

                for i, section in enumerate(sections):
                    print(f"Processing section {i+1} of {len(sections)}...")
                    summary = summarize_text(section)
                    if summary:
                        all_summaries.append(summary)

                    insights = extract_insights(section)
                    if insights:
                        all_insights.append(insights)

                final_summary = " ".join(all_summaries)
                final_insights = " ".join(all_insights)

                # Generate network graph (no display in web app, save to file)
                G = create_network_graph(extracted_data)  # Function defined below
                nt = Network('1000px', '2000px', notebook=False) # notebook=False for saving to file
                nt.from_nx(G)
                graph_filename = f"{filename[:-4]}_graph.html" # Name based on PDF filename
                nt.save_graph(os.path.join(app.config['UPLOAD_FOLDER'], graph_filename))


                return render_template('results.html', summary=final_summary, insights=final_insights, graph_filename=graph_filename)

            except Exception as e:
                return f"An error occurred: {e}"
        else:
            return "Invalid file type. Please upload a PDF."

    return render_template('index.html')

def create_network_graph(extracted_data): # Function to create the graph
    document_dict = extracted_data['data']
    G = nx.DiGraph()
    G.add_node(1, label=extracted_data['name'], title=extracted_data['name'], color="black")

    for i in range(len(document_dict)):
        # ... (rest of the graph creation code, same as before)
        page_idx = i + 2
        page_num = "Page_" + str(document_dict[i]['page'])
        G.add_node(page_idx, label=page_num, title=page_num, color="blue")
        G.add_edge(1, page_idx)

        text_idx = 'text_' + str(i + 1)
        G.add_node(text_idx, label=document_dict[i]['text'][:100] + "...", title=document_dict[i]['text'], color="violet")
        G.add_edge(page_idx, text_idx)

        table_flag_idx = 'table_flag' + str(i + 1)
        has_tables = 1 if document_dict[i]['tables'] else 0
        G.add_node(table_flag_idx, label=str(has_tables), title=str(has_tables), color="green")
        G.add_edge(page_idx, table_flag_idx)

        for table_index, table in enumerate(document_dict[i]['tables']):
            table_idx = f'table_{i+1}_{table_index}'
            table_label = "Table " + str(table_index + 1)
            G.add_node(table_idx, label=table_label, title=table.to_string(), color="orange")
            G.add_edge(table_flag_idx, table_idx)


        images_list = document_dict[i].get('img_npy_lst', [])

        for image_index, image_metadata in enumerate(images_list):
            image_idx = f'image_{i + 1}_{image_index}'
            image_label = "Image " + str(image_index + 1)
            G.add_node(image_idx, label=image_label, title=str(image_metadata), color="purple")
            G.add_edge(page_idx, image_idx)
    return G


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
