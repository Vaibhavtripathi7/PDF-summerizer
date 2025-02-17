import google.generativeai as genai
import pickle

def preprocess_text(text):
    text = " ".join(text.split())
    return text

genai.configure(api_key="AIzaSyCtjwJEDBUttvJ_dnjjCG8WHv98nVGYWLE")

model = genai.GenerativeModel('gemini-pro')

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

if __name__ == "__main__":
    try:
        with open('pdf_content_dict_stage1.pickle', 'rb') as handle:
            extracted_data = pickle.load(handle)

        combined_text = combine_text_from_pdf(extracted_data)

        if not combined_text.strip():
            print("Error: No text extracted from PDF. Check the pickle file.")
            exit()

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

        print("\n--- Summary ---")
        print(final_summary)

        print("\n--- Insights ---")
        print(final_insights)




    except FileNotFoundError:
        print("Error: 'pdf_content_dict_stage1.pickle' not found. Make sure it exists.")
    except pickle.UnpicklingError:
        print("Error: Could not load data from the pickle file. It might be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")