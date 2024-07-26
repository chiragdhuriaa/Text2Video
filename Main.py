import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import imageio
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from huggingface_hub import snapshot_download
import pathlib
import numpy as np

app = tk.Tk()
app.geometry("532x632")
app.title("Text2Video Synthesizer")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

model_dir = pathlib.Path('weights')
snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis', repo_type='model', local_dir=model_dir)
pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())

def is_nsfw_prompt(prompt_text):
    nsfw_keywords = ["nude", "nudity", "sex", "porn", "explicit", "nsfw"]
    return any(keyword in prompt_text.lower() for keyword in nsfw_keywords)

def generate():
    prompt_text = prompt.get()
    if is_nsfw_prompt(prompt_text):
        print("NSFW content detected in the prompt. Please use a different prompt.")
        return

    try:
        test_text = {'text': prompt_text}
        output_video_path = pipe(test_text)[OutputKeys.OUTPUT_VIDEO]
        print('output_video_path:', output_video_path)

        video_capture = imageio.get_reader(output_video_path)
        for frame in video_capture:
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            lmain.configure(image=img)
            lmain.image = img  
            lmain.update()
    except Exception as e:
        print(f"Error generating video: {e}")

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
