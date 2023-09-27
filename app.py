import os
from typing import Optional, Tuple
from threading import Lock

import gradio as gr

from query_data import get_basic_qa_chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    api_key = os.getenv("OPENAI_API_KEY")  # 讀取環境變數中的API金鑰
    print(api_key)
    if api_key:
        chain = get_basic_qa_chain()
        return chain
    #if api_key:
     #   os.environ["OPENAI_API_KEY"] = api_key
      #  chain = get_basic_qa_chain()
       # os.environ["OPENAI_API_KEY"] = ""
        #return chain


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
      

    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                 chain = get_basic_qa_chain()
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain({"question": inp})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}",title="政府採購法客服機器人")

with block:
   # with gr.Row():
    #    gr.Markdown(
     #       "<h3><center>Chat-Your-Data (State-of-the-Union)</center></h3>")

      #  openai_api_key_textbox = gr.Textbox(
       #     placeholder="Paste your OpenAI API key (sk-...)",
        #    show_label=False,
         #   lines=1,
          #  type="password",
        #)

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="你想問什麼採購相關問題?",
            placeholder="有關政府採購法相關問題都可以問",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(
            full_width=False)

    gr.Examples(
        examples=[
            "決標要注意什麼?",
            "什麼是勞務採購?",
            "得標廠商於決標後稱其標價書寫錯誤拒不簽約，押標金會不會被没收"
        ],
        inputs=message,
    )

    gr.HTML("，　<center><a href ='https://www.pcc.gov.tw/cp.aspx?n=41570175D3B41EEE'>採購法常見問答系統，資料來源為工程會常見問答集資料</a> </center>")

    #gr.HTML(
#     #   "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
  #  )

    state = gr.State()
    agent_state = gr.State()
    
   # submit.click(chat, inputs=[openai_api_key_textbox, message,
    submit.click(chat, inputs=[message, message,
    #apikey=os.getenv("OPENAI_API_KEY")
    #submit.click(chat, inputs=[apikey , message,
                 state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[
                #   openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
                  message, message, state, agent_state], outputs=[chatbot, state])

    #openai_api_key_textbox.change(
     ##  inputs=[openai_api_key_textbox],
       # outputs=[agent_state],
    #)

#block.launch(debug=True)
block.launch(share=False,server_name="0.0.0.0")
