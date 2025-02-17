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


doc_path = "/home/vaibhav/Downloads/Enhancing_Real-time_Object_Detection_with_YOLO_Alg-2.pdf"
extracted_data = extract_text_images_tables(doc_path, download=True)

import pickle

with open('pdf_content_dict_stage1.pickle', 'wb') as handle:
    pickle.dump(extracted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol for compatibility

with open('pdf_content_dict_stage1.pickle', 'rb') as handle:
    loaded_extracted_data = pickle.load(handle)



document_dict = extracted_data['data'] #Use the enriched data here
doc_len = len(document_dict) + 1

G = nx.DiGraph()
G.add_node(1, label=extracted_data['name'], title=extracted_data['name'], color="black")

pages_with_no_tables = []
pages_with_no_images = []

for i in range(len(document_dict)):
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


    images_list = document_dict[i].get('images', [])

    for image_index, image_metadata in enumerate(images_list):
        image_idx = f'image_{i + 1}_{image_index}'
        image_label = "Image " + str(image_index + 1)
        G.add_node(image_idx, label=image_label, title=str(image_metadata), color="purple")
        G.add_edge(page_idx, image_idx)


nt = Network('1000px', '2000px', notebook=True)
nt.from_nx(G)
nt.toggle_physics(True)
nt.show('nx1.html')

