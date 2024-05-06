import time
from time import sleep

from llm_api import *

import customtkinter as ctk

from utilities import insert_newlines


class PromptHistory:
    def __init__(self, app):
        self._max_line_length = 20

        self._prompt_history = []

        self.create_text_area(app)

    def create_text_area(self, app):
        label = ctk.CTkLabel(app, text="A demo of LLM tool usage", font=("Helvetica", 30))#, fg_color="transparent")
        label.place(relx=0.35, rely=0.05, anchor=ctk.N)

        # Scrolling text area
        self.text_area = ctk.CTkTextbox(app, width=400, wrap="word", state='disabled', font=font)

        self.text_area.grid(row=0, column=0, sticky="nsew")
        self.text_area.place(relx=0.5, rely=0.35, relwidth=0.8, relheight=0.45, anchor=ctk.CENTER)

    def add_prompt(self, prompt, history=None):
        self._prompt_history.append(prompt)

        prefix = "" 
        if type(prompt) == SystemMessage:
            prefix = "System"
            return
        elif type(prompt) == HumanMessage:
            prefix = "User"
        elif type(prompt) == AIMessage:
            prefix = "AI"

        try: 
            prompt = prompt.content
        except:
            pass

        self.text_area.configure(state='normal')
        text = self.text_area.get("0.0", "end")  # get text from line 0 character 0 till the end
        self.text_area.delete("0.0", "end")  # delete all text
        print(text, len(text))
        if len(text) > 1:
            new_line = f"{text}\n{prefix}: {prompt}"
        else:
            new_line = f"{prefix}: {prompt}"
        self.text_area.insert("0.0", new_line)  # insert at line 0 character 0

        self.text_area.see("end")  # scroll to the end

        self.text_area.configure(state='disabled')

class PromptIdeas:
    def __init__(self, app):
        self.prompts = ["What is 16 * 256 and is it a prime?", "What are some of the current Norwegian news?", "What is the weather in Oslo, Norway right now?"]
        self._max_line_length = 16

        self._callback = None
        self._buttons = []

        self.create_buttons(app)

    def create_buttons(self, app):
        for i, prompt in enumerate(self.prompts):
            button = ctk.CTkButton(app, text=insert_newlines(prompt, self._max_line_length), command=self.button_callback(i), font=font, width=170, height=100)
            button.place(relx=0.2 + 0.3 * i, rely=0.7, anchor=ctk.CENTER)
            self._buttons.append(button)

    def button_callback(self, i):
        def func():
            print(i)
            if self._callback is not None:
                prompt = self.prompts[i]
                self._callback(prompt)

        return func

    def set_callback(self, callback):
        self._callback = callback

    def update_prompts(self, new_message, history):
        if type(new_message) != AIMessage:
            return

        start_time = time.time()

        new_prompts = set()
        for i in range(3):
            messages = [system_idea]
            messages.extend(history[1:])

            response = idea_chain.invoke(messages)
            print(response)

            t = list(map(lambda x: x["args"]["prompt"], response))

            new_prompts.update(t)

            print(new_prompts)
            
            if len(new_prompts) >= 3:
                break

            print("Didnt get 3 prompts, trying again")
            sleep(0.1)

        if len(new_prompts) < 3:
            print("Failed to get 3 prompts")
            return
        
        self.prompts = list(new_prompts)

        for i, prompt in enumerate(self.prompts):
            self._buttons[i].configure(text=insert_newlines(prompt, self._max_line_length))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

class PromptField:
    def __init__(self, app):
        self._max_line_length = 20

        self._create_input_field(app)

        self._callback = None

    def _create_input_field(self, app):
        # Input field
        self.input_field = ctk.CTkEntry(app, font=font, placeholder_text="Ask any question")
        self.input_field.place(relx=0.45, rely=0.85, relwidth=0.7, anchor=ctk.CENTER)
        self.input_field.bind('<Return>', self._submitted_callback)

        # Submit button
        submit_button = ctk.CTkButton(app, text="Ask!", width=20, font=font, command=self._submitted_callback)
        submit_button.place(relx=0.85, rely=0.85, anchor=ctk.CENTER)

    def set_callback(self, callback):
        self._callback = callback   

    def _submitted_callback(self, event=None):
        print(event)
        if self._callback is not None and self.input_field.get() != '':
            self._callback(self.input_field.get())
            self.input_field.delete(0, 'end')


class LLMLogic:
    def __init__(self, llm_tools, llm_user_reponse):
        self._prompt_history = []
        self._llm_tools = llm_tools
        self._llm_user_reponse = llm_user_reponse

        self._prompt_history.append(system)

        self._callback_new_message = []

    def process_prompt(self, prompt):
        human = HumanMessage(
            content=prompt
        )
        self._prompt_history.append(human)

        for callback in self._callback_new_message:
            callback(human, history=self._prompt_history)

        tool_reponse = self._llm_tools.invoke({"history": self._prompt_history})#, functions=self.functions)

        print(tool_reponse) 
        if len(tool_reponse["tools"]) != 0:
            self._prompt_history.append(tool_reponse["ai_message"])
            self._prompt_history.extend(tool_reponse["tools"])

            ai_message = self._llm_user_reponse.invoke({"history": self._prompt_history})
        else:
            ai_message = tool_reponse["ai_message"]
        print(ai_message)

        self._prompt_history.append(ai_message) 

        for callback in self._callback_new_message:
            x = threading.Thread(target=callback, args=(ai_message, self._prompt_history))
            x.start()

    def add_callback_new_message(self, callback):
        self._callback_new_message.append(callback)


ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.title("My awesome LLM GUI")
app.geometry("800x800")
app.resizable(False, False)

font = ("Helvetica", 18)

prompt_history = PromptHistory(app)
prompt_ideas = PromptIdeas(app)
prompt_field = PromptField(app)

llm_logic = LLMLogic(tool_chain, response_chain)

prompt_field.set_callback(llm_logic.process_prompt)
prompt_ideas.set_callback(llm_logic.process_prompt)

llm_logic.add_callback_new_message(prompt_history.add_prompt)
llm_logic.add_callback_new_message(prompt_ideas.update_prompts)

app.mainloop()