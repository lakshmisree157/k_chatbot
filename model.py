import whisper
import json
import os

# Load Whisper model
model = whisper.load_model("base")  # Adjust model size as needed

# Folder containing your audio files
audio_folder = "C:\\Users\\DELL\\ml_data"
output_json = "transcriptions.json"

# Prepare an array to store all transcriptions
transcriptions = {"audio_responses": []}

# Loop through all files in the audio folder
for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3"):  # Process only mp3 files, adjust if needed
        audio_path = os.path.join(audio_folder, filename)
        
        # Transcribe and translate to English if needed
        result = model.transcribe(audio_path, language="kn", task="translate")
        
        # Store the transcription and file information
        transcriptions["audio_responses"].append({
            "id": filename.replace(".mp3", ""),
            "text": result["text"],
            "file": filename
        })

# Save all transcriptions to a JSON file
with open(output_json, "w", encoding="utf-8") as json_file:
    json.dump(transcriptions, json_file, ensure_ascii=False, indent=4)

print(f"Transcriptions saved to {output_json}")
import tkinter as tk

from sentence_transformers import SentenceTransformer, util
import whisper

# Load the Whisper model for Kannada ASR (adjust model size as needed)
asr_model = whisper.load_model("small")

# Load JSON data (ensure the file path is correct)
json_file_path = r"C:\Users\DELL\ml_data\transcriptions.json"
with open(json_file_path, 'r', encoding='utf-8') as file:
    transcriptions_dict = json.load(file)

# Load sentence-transformers model for matching questions to responses
model = SentenceTransformer('all-MiniLM-L6-v2')
questions = list(transcriptions_dict.keys())
question_embeddings = model.encode(questions)

def find_best_response(user_input):
    # Find the best matching response from JSON data
    user_embedding = model.encode(user_input)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)
    best_match_idx = similarities.argmax()
    best_question = questions[best_match_idx]
    best_answer = transcriptions_dict[best_question]
    return best_answer

# Main function for Kannada audio transcription
def transcribe_kannada_audio(audio_path):
    result = asr_model.transcribe(audio_path)
    return result["text"]

# Tkinter Application
root = tk.Tk()
root.title("Kannada to English Chatbot")
root.geometry("400x300")
root.configure(bg="#AEDFF7")

# Style Functions
def apply_styles(widget, font_size=12, bg_color="#007ACC", fg_color="white"):
    widget.configure(font=("Arial", font_size), bg=bg_color, fg=fg_color)

# Main Menu to choose input type
def main_menu():
    for widget in root.winfo_children():
        widget.destroy()
        
    tk.Label(root, text="Choose Input Type", font=("Arial", 16, "bold"), bg="#AEDFF7").pack(pady=20)
    
    text_button = tk.Button(root, text="English Text Input", command=open_text_input)
    apply_styles(text_button, font_size=14)
    text_button.pack(pady=10)
    
    voice_button = tk.Button(root, text="Kannada Voice Input", command=open_voice_input)
    apply_styles(voice_button, font_size=14)
    voice_button.pack(pady=10)

# Text Input Window
def open_text_input():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Enter your question in English:", font=("Arial", 12), bg="#AEDFF7").pack(pady=5)
    entry_question = tk.Entry(root, width=40)
    entry_question.pack(pady=10)

    output_text = scrolledtext.ScrolledText(root, width=50, height=10)
    output_text.pack(pady=10)
    
    def on_submit():
        question = entry_question.get()
        if question.strip():
            answer = find_best_response(question)
            output_text.insert(tk.END, "You: " + question + "\n")
            output_text.insert(tk.END, "Chatbot: " + answer + "\n\n")
            entry_question.delete(0, tk.END)
            
    submit_button = tk.Button(root, text="Get Response", command=on_submit)
    apply_styles(submit_button, font_size=12)
    submit_button.pack(pady=5)

    back_button = tk.Button(root, text="Back to Main Menu", command=main_menu)
    apply_styles(back_button, font_size=12, bg_color="#F39C12")
    back_button.pack(pady=10)

# Voice Input Window
def open_voice_input():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Upload Kannada Audio File", font=("Arial", 12), bg="#AEDFF7").pack(pady=10)
    
    output_text = scrolledtext.ScrolledText(root, width=50, height=10)
    output_text.pack(pady=10)

    def on_audio_submit():
        audio_path = tk.filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if audio_path:
            transcription = transcribe_kannada_audio(audio_path)
            answer = find_best_response(transcription)
            output_text.insert(tk.END, "Transcription: " + transcription + "\n")
            output_text.insert(tk.END, "Chatbot: " + answer + "\n\n")

    upload_button = tk.Button(root, text="Upload Audio and Get Response", command=on_audio_submit)
    apply_styles(upload_button, font_size=12)
    upload_button.pack(pady=10)

    back_button = tk.Button(root, text="Back to Main Menu", command=main_menu)
    apply_styles(back_button, font_size=12, bg_color="#F39C12")
    back_button.pack(pady=10)

# Run the main menu on start
main_menu()
root.mainloop()
