#!/usr/bin/python
# -*- coding: utf-8 -*-


# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from constants import *

def btn_choosedata():
	global var_dataDir
	
	var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
	button_run.configure(state=NORMAL)
	button_choose.configure(state=DISABLED)
	button_load.configure(state=NORMAL)
        
        
def btn_chooseobjectdata():
	
        
	
	#var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
        file_object_path = askopenfilename()
        
        entry_object.delete(0, END)
        entry_object.insert(0, file_object_path)
        
        if entry_object.get()!="" and entry_image.get()!="" and entry_point.get()!="":
            button_load.configure(state=NORMAL)

	
	
        
        
def btn_chooseimagedata():
	#global var_dataDir
	
	#var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
        file_image_path = askopenfilename()
        
        entry_image.delete(0, END)
        entry_image.insert(0, file_image_path)
        

        if entry_object.get()!="" and entry_image.get()!="" and entry_point.get()!="":
            button_load.configure(state=NORMAL)
        
def btn_choosepointdata():
	#global var_dataDir
	
	#var_dataDir = askdirectory(initialdir='/Users/benjamin/dartigues/super_class_test/')
	file_point_path = askopenfilename()
        
        entry_point.delete(0, END)
        entry_point.insert(0, file_point_path)

        if entry_object.get()!="" and entry_image.get()!="" and entry_point.get()!="":
            button_load.configure(state=NORMAL)


def btn_load_data():

        #button_load.configure(state=DISABLED)


	launch_experiment(
		classification_mode,
		per_image_file,
                per_image_cols,
		per_object_file, 
                per_object_cols,
		per_point_file,
                per_point_cols,
		groups)


#old main for interface
if __name__ == "__main__":
    
        #global file_object_path, file_image_path,file_point_path

        
	winDataChoose = Tk()
	winDataChoose.title('Projet "SuperClass" - Cell Classification Algorithm (v0)')
	winDataChoose.focus_force()
        file_object_path= StringVar(winDataChoose)
        file_point_path = StringVar(winDataChoose)
        file_image_path= StringVar(winDataChoose)
        
        
        menubar = Menu(winDataChoose)
        winDataChoose.config(menu=menubar)
        filemenu = Menu(menubar)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Close", command=winDataChoose.quit)
        
        
        
        #label1 = Label(winDataChoose, text='Choose the directory of data, to run the classification algorithm on it: ')
	#label1.grid(row=0, column=0, pady=20, padx = 0)
	#button_choose = Button(winDataChoose, text='Choose Object files...', command = btn_choosedata)
	#button_choose.grid(row=0, column=1, pady=20, padx = 0) 
	
	
	
	samples=[]
	#var_dataDir = StringVar(winDataChoose)
	#var_dataDir.set('')
	 # Additionale file provided by Anne
         
        
        
         

        labelobjet = Label(winDataChoose, text='Choose image file, to run the classification algorithm on it: ')
	labelobjet.grid(row=0, column=0, pady=20, padx = 0)
	button_object_choose = Button(winDataChoose, text='Choose Object files...', command = btn_chooseobjectdata)
	button_object_choose.grid(row=0, column=1,sticky='ew', pady=20, padx = 0)
        entry_object = Entry(winDataChoose, width=50, textvariable=file_object_path)
        entry_object.grid(row=0,column=2,padx=2,pady=2,sticky='we',columnspan=25)
        
        #Label(winDataChoose,text="File name: '").grid(row=5, column=0, sticky='e')
        #entry = Entry(winDataChoose, width=50, textvariable=file_object_path)
        #entry.grid(row=5,column=1,padx=2,pady=2,sticky='we',columnspan=25)
        #Button(winDataChoose, text="Browse", command=btn_chooseobjectdata).grid(row=1, column=27, sticky='ew', padx=8, pady=4)
        
        
        
        
        labelimage= Label(winDataChoose, text='Choose object file, to run the classification algorithm on it: ')
	labelimage.grid(row=1, column=0, pady=20, padx = 0)
        button_image_choose = Button(winDataChoose, text='Choose Images files...', command = btn_chooseimagedata)
	button_image_choose.grid(row=1, column=1, pady=20, padx = 0)
        entry_image = Entry(winDataChoose, width=50, textvariable=file_image_path)
        entry_image.grid(row=1,column=2,padx=2,pady=2,sticky='we',columnspan=25)
        
        
        labelpoint = Label(winDataChoose, text='Choose points file, to run the classification algorithm on it: ')
	labelpoint.grid(row=2, column=0, pady=20, padx = 0)
        button_point_choose = Button(winDataChoose, text='Choose points files...', command = btn_choosepointdata)
	button_point_choose.grid(row=2, column=1, pady=20, padx = 0)
        entry_point = Entry(winDataChoose, width=50, textvariable=file_point_path)
        entry_point.grid(row=2,column=2,padx=2,pady=2,sticky='we',columnspan=25)
        
        
        Label(winDataChoose, text='Load data, to run the classification algorithm on it: ').grid(row=3, column=0, pady=20, padx = 0)
	button_load = Button(winDataChoose, text='Load ...', command = btn_load_data)
	button_load.grid(row=3, column=1, pady=20, padx = 0)
	#button_load.configure(state=DISABLED)


	#button_run = Button(winDataChoose, text='Run Cell Classification Algorithm (v0)', command=btn_runCellClassificationAlgo)
	#button_run.configure(state=DISABLED)
	#button_run.grid(row=4, column=0, columnspan=2)
        
        #button_unsupervised_run = Button(winDataChoose, text='Run Unsupervised Cell Classification Algorithm (v0)', command=run_unsupervised_cell_classification_algo)
	#button_unsupervised_run.configure(state=DISABLED)
	#button_unsupervised_run.grid(row=5, column=0, columnspan=2)

	label2 = Label(winDataChoose, text='Show result graphs: ')
	label2.grid(row=6, column=0, sticky=E)

	var_showGraphs = StringVar(winDataChoose)
	var_showGraphs.set("No") # initial value
	option_showGraphs = OptionMenu(winDataChoose, var_showGraphs, "No", "Yes")
	option_showGraphs.grid(row=6, column=1, sticky=W)



       
        
        
	canvas_results = Canvas(winDataChoose, width=550, height=400, bd=1,relief='ridge')
	canvas_results.grid(row=6, column=0, columnspan=2)
        
        btn_load_data()

#        objectcsvfile=Label(root, text="ObjectFile").grid(row=0, column=2)
#        imagecsvfile=Label(root, text="ImageFile").grid(row=1, column=2)
#        pointcsvfile=Label(root, text="PointFile").grid(row=2, column=2)
#        bar=Entry(master).grid(row=1, column=1) 
#	winDataChoose.mainloop()