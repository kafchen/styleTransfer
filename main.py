from tkinter.filedialog import askopenfilename
import os
import tkinter as tk
from tkinter import *  # 图形界面库
import tkinter.messagebox as messagebox  # 弹窗
from ttkbootstrap import Style
from PIL import Image, ImageTk
import styleTrans
import threading
import queue


class re_Text():

    def __init__(self, queue):
        self.queue = queue

    def write(self, content):
        self.queue.put(content)


class stPage:
    def __init__(self,root):
        self.msg_queue = queue.Queue()
        self.initGUI(root)

    def show_msg(self):

        while not self.msg_queue.empty():
            content = self.msg_queue.get()
            self.text.insert(INSERT, content)
            self.text.see(END)
        # after方法再次调用show_msg
        self.window.after(1000, self.show_msg)



    def initGUI(self,root):
        self.style = Style(theme='lumen')
        self.window = root
        self.window = self.style.master
        self.window.title('StyleTransfer')
        self.window.geometry('900x750+250+25')
        self.window.resizable = False
        self.msg_queue = queue.Queue()
        self.content_path = StringVar()
        self.style_path = StringVar()
        self.content_path.set(os.path.abspath("."))
        self.style_path.set(os.path.abspath("."))

        self.content_input = tk.Frame(self.window, bg='white', width=400, height=35, relief='groove', bd=3)
        self.style_input = tk.Frame(self.window, bg='white', width=400, height=35, relief='groove', bd=3)
        self.content_pic = tk.Frame(self.window, bg='white', width=400, height=300, relief='groove', bd=3)
        self.style_pic = tk.Frame(self.window, bg='white', width=400, height=300, relief='groove', bd=3)
        self.resulttext = tk.Frame(self.window, width=400, height=60)
        self.result = tk.Frame(self.window, bg='white', width=400, height=300, relief='groove', bd=3)
        self.set = tk.Frame(self.window, width=400, height=60)
        self.log = tk.Frame(self.window, bg='white', width=400, height=300, relief='groove', bd=3)

        self.content_input.grid(row=0,column=0,padx=15,pady=3)
        self.style_input.grid(row=0, column=1, padx=30, pady=3)
        self.content_pic.grid(row=1, column=0, padx=15, pady=3)
        self.style_pic.grid(row=1, column=1, padx=30, pady=3)
        self.resulttext.grid(row=2, column=1, padx=30, pady=5)
        self.result.grid(row=3, column=1, padx=30, pady=0)
        self.set.grid(row=2, column=0, padx=15, pady=5)
        self.log.grid(row=3, column=0, padx=15, pady=5)

        self.scrollBar = Scrollbar(self.log)
        self.scrollBar.pack(side="right", fill="y")

        self.text = Text(self.log, width=40, height=15,yscrollcommand=self.scrollBar.set)
        self.text.pack(side="top")
        self.scrollBar.config(command=self.text.yview)




        Label(self.content_input, text="主图片:").pack(anchor=NW,side=LEFT,padx=5,pady=2)
        Entry(self.content_input, textvariable=self.content_path,width=38).pack(anchor=NW,side=LEFT,padx=5,pady=2)
        Button(self.content_input, text="选择路径", command=lambda:self.selectPath(self.content_path,1)).pack(anchor=NW,side=LEFT,padx=0,pady=0)

        Label(self.style_input, text="风格图片:").pack(anchor=NW, side=LEFT, padx=5, pady=2)
        Entry(self.style_input, textvariable=self.style_path, width=38).pack(anchor=NW, side=LEFT, padx=5, pady=2)
        Button(self.style_input, text="选择路径", command=lambda: self.selectPath(self.style_path, 0)).pack(anchor=NW,side=LEFT,padx=0, pady=0)


        Label(self.resulttext, text="风格迁移结果",font=('微软雅黑',15)).pack(anchor=S, side=BOTTOM, padx=5, pady=0)

        self.epoch = StringVar(value='10')
        Label(self.set, text="输入你要训练的轮次:").grid(row=0,column=0,padx=2,pady=1)
        Entry(self.set, textvariable=self.epoch,width=10).grid(row=0,column=1,padx=2,pady=1)
        self.style_weight = StringVar(value='0.01')
        Label(self.set, text="输入风格权重:").grid(row=1,column=0,padx=2,pady=1)
        Entry(self.set, textvariable=self.style_weight, width=10).grid(row=1,column=1,padx=2,pady=1)
        self.content_weight = StringVar(value='1000')
        Label(self.set, text="输入内容权重:").grid(row=2,column=0,padx=2,pady=1)
        Entry(self.set, textvariable=self.content_weight, width=10).grid(row=2,column=1,padx=2,pady=1)
        Button(self.set, text="开始迁移", command=lambda:self.styleTrans()).grid(row=1,column=3,padx=15,pady=0)

        self.window.after(1000, self.show_msg)

        sys.stdout = re_Text(self.msg_queue)

        self.window.mainloop()  # 主消息循环

    def selectPath(self,path,flag):
        path_ = askopenfilename()  # 使用askdirectory()方法返回文件夹的路径
        if path_ == "":
            path.get()  # 当打开文件路径选择框后点击"取消" 输入框会清空路径，所以使用get()方法再获取一次路径
        else:
            path_ = path_.replace("/", "\\")  # 实际在代码中执行的路径为“\“ 所以替换一下
            path.set(path_)
            if '.jpg' or '.png' in path_:
                if flag == 1:
                    self.show_contentPic(path.get())
                else:
                    self.show_stylePic(path.get())


    def show_contentPic(self,path):
        global contentPic
        img = Image.open(path)
        img = img.resize((400,300))
        contentPic = ImageTk.PhotoImage(img)
        Label(self.content_pic, image=contentPic, width=400, height=300).pack()

    def show_stylePic(self,path):
        global stylePic
        img = Image.open(path)
        img = img.resize((400,300))
        stylePic = ImageTk.PhotoImage(img)
        Label(self.style_pic, image=stylePic, width=400, height=300).pack()


    def __styleTrans(self):
        content_path = self.content_path.get()
        style_path = self.style_path.get()
        st = styleTrans.styleTransfer(content_path, style_path, int(self.epoch.get()),float(self.style_weight.get()), int(self.content_weight.get()))
        st.getResult()

        global result
        img = Image.open('stylized-image11.png')
        img = img.resize((400, 300))
        result = ImageTk.PhotoImage(img)
        Label(self.result, image=result, width=400, height=300).pack()

    def styleTrans(self):
        T = threading.Thread(target=self.__styleTrans(), args=())
        T.setDaemon(True)#守护线程
        T.start()



if __name__ == '__main__':
    try:
        root = Tk()
        stPage(root)
    except:
        messagebox.showinfo('错误！', '请重新操作')


