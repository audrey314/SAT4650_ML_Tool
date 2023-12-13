import random
import tkinter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tkinter import messagebox as tkMessageBox
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
import matplotlib.pyplot as plt
from functools import partial
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np


big_font = ("Arial", 16)
med_font = ("Arial", 12)

title_size = 12
plt.rcParams.update({'font.size':12})
plt.rcParams['axes.titlepad'] = 10


all_model_options = {
            'K Nearest Neighbors':[KNeighborsClassifier,{"# of Neighbors":["n_neighbors",[1, 4, 7, 10, 13, 16, 19, 22, 25]]}],
            'Support Vector Machine':[svm.SVC,{"Cost":["C",[0.01,0.1,1,10,100]],"Kernel":["kernel",["rbf","linear"]]}],
            'Decision Tree' : [tree.DecisionTreeClassifier,{"Max Depth":["max_depth",[2,5,10,15,20,25]]}]
        }

helpful_text = {
    'seed':["What is a random seed?","A random seed is a number used to determine the outcome of random decisions in the model building process. Changing the random seed allows you to see how the models may vary due to randomness. Leaving the random seed at a set value allows you to compare multiple models with the same data and to get repeatable results if you run the model multiple times."],
    'train':["What is a train/test split?",  "When building machine learning models it is important to train on a training data set and evaluate the performance of the models on different data that wasn't used to train the model. It is possible to overfit and 'memorize' the data which would give good results on the training set but bad results if used on any other data. The test set gives an idea of how the model might perform actual real world applications. "],
    'center':["What does it mean to center the data","Centering the data is done by subtracting the mean of each column from each column The result of centering is that each column will have a mean of 0. Centering is also a way to help reduce multicollinearity in regression models. "],
    'scale':["What does it mean to scale the data" ,"Scaling the data is done by dividing each column by the standard deviation of each column. The result of scaling is that each column will have a standard deviation of 1. Scaling is helpful because it prevents columns of large values from having more weight in the model than columns of small values."],
    'model':["What are the different models?","Explain each model and parameters"],
    'n_neighbors':["What is K?", "In K Nearest Neighbors classification, it labels a data point based on the k closest points to that data point. The value of k is a tuning parameter - we can pick if we want to use the 1 closest neighbor, 2 closest neighbors, and so on. Typically, it is best to use values of k that will avoid ties. If there are two classes it is best to use odd values of k."],
    'C':["What is the Cost parameter","The cost parameter for a Support Vector Machine is used to give a penalty weight to misclassified data points during training. Larger values of cost tend to overfit the data and smaller values of cost tend to underfit the data."],
    'kernel':["What is a kernel?","The Support Vector Machine uses kernels, which are mathematical functions used to try and separate out the different classes. The linear kernel uses linear functions and the rbf kernel uses radial basis functions which are nonlinear functions."],
    'max_depth':["What does max depth mean?","Decision Trees determine criteria to split the data into groups with the goal of separating out all of the classes. The max depth of the decision tree is the maximum number of times the decision tree can split the data. Smaller values of max_depth tend to underfit the data while larger values of max depth tend to overfit the data"]

}


class myGUI:

    #make dropdown from dictionary or list
    def add_dropdown(self, dictlist, frame, callback_func,wid=20,padx=5):
        tkvar = tkinter.StringVar(self.main_window)
        if isinstance(dictlist,dict):
            dropdown = tkinter.OptionMenu(frame,tkvar,
                                      command=callback_func,
                                    *set(dictlist.keys()))
        elif isinstance(dictlist,list):
            dropdown = tkinter.OptionMenu(frame, tkvar,
                                          command=callback_func,
                                          *dictlist)

        dropdown.config(font=med_font, bg='white', width=wid)
        dropdown.nametowidget(dropdown.menuname).config(font=med_font,background="white")
        dropdown.pack(padx=padx)

    # helper function for other callback functions
    # takes a dictionary and args which is a keyword
    # returns name and dictionary stored in the dictionary at the keyword
    def change_helper(self,dict,*args):
        key=args[0]
        dict_val = dict[key]
        name=dict_val[0]
        new_dict=dict_val[1]
        return name,new_dict


    # Callback function for when the user selects a different model
    def model_change(self,*args):
        #clear out previously stored params
        self.params={}
        name, new_dict = self.change_helper(all_model_options,*args)
        self.model=name

        # remove previous param options when new model is selected
        self.inner_frame.destroy()
        self.inner_frame = tkinter.Frame(self.param_frame,background="white")
        self.inner_label = tkinter.Label(self.inner_frame, text="Select Parameters",font=med_font,background="white",pady=3)
        self.inner_label.pack()
        self.inner_frame.pack()

        #store list of parameters names
        self.param_list=[]
        keys = list(new_dict.keys())

        # Iterate through the parameters creating new dropdowns
        for i in range(len(new_dict.keys())):
            k=keys[i]
            temp_frame = tkinter.Frame(self.inner_frame)
            name, list_vals = self.change_helper(new_dict,k,*args)

            self.make_info_button(temp_frame,"left",name)

            label = tkinter.Label(temp_frame,text=k,font=med_font,background="white")
            label.pack(side="left",padx=4)

            self.param_list.append(name)
            self.add_dropdown(list_vals, temp_frame, partial(self.param_change,self.param_list[i]),wid=6)
            temp_frame.config(background="white",pady=5)
            temp_frame.pack()
        self.inner_frame.pack()

    #adds parameters to the dictionary to be accessed later when building the model
    def param_change(self, param, *args):
        self.params[param]=args[0]

    def tr_te_split(self,*args):
        test_size = 1-args[0]/100
        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X[:,[0,2]], y, test_size=test_size, random_state=self.seed)

    def change_seed(self,*args):
        self.seed = random.randint(0,1000)
        self.rand_seed_lb.config(text=f"Random Seed: {self.seed}",font=med_font,pady=10,padx=5)

    def disp_info(self,vals):
        title_val, msg_val = vals
        tkMessageBox.showinfo(title_val,msg_val)

    def make_info_button(self,frame,side,kw):
        info_button = tkinter.Button(frame, text="?", font=med_font,
                                     command=partial(self.disp_info, helpful_text[kw]))
        info_button.pack(side=side)

    # make configured frame for organization purposes
    def make_temp_frame(self,frame):
        temp_frame = tkinter.Frame(frame)
        temp_frame.config(background="white", pady=5)
        temp_frame.pack()
        return temp_frame

    def __init__(self):
        #set initial random seed
        self.seed = random.randint(0,1000)

        # list of named variables for later use
        self.params = {}
        self.param_list = []
        self.model=None
        self.modelfit=None
        self.predicted=None
        self.X_train = None

        # create main window widget
        self.main_window = tkinter.Tk()
        self.main_window.config(background="white",padx=5,pady=5)
        self.main_window.rowconfigure(3,minsize=50)
        self.main_window.columnconfigure(10, minsize=30)

        # create frame for data preprocessing
        self.preproc_frame = tkinter.Frame(self.main_window)
        self.preproc_frame.pack(pady=3,padx=10)
        self.preproc_frame.config(background="white")
        self.lb_preproc = tkinter.Label(self.preproc_frame, text = "Preprocessing",font=big_font,background="white")
        self.lb_preproc.pack()

        #create temp frame for organization of buttons and labels inside the preproc frame
        temp_frame = self.make_temp_frame(self.preproc_frame)
        self.make_info_button(temp_frame,"left",'seed')
        self.rand_seed_lb = tkinter.Label(temp_frame, text=f"Random Seed: {self.seed}",font=med_font,background="white",padx=5,pady=10)
        self.rand_seed_lb.pack(side="left")
        self.rand_seed_button = tkinter.Button(temp_frame,text="New Seed",command = self.change_seed,font=med_font, pady=1,padx=3)
        self.rand_seed_button.pack(side="right")

        #create temp frame for organization of buttons and labels inside the preproc frame
        temp_frame = self.make_temp_frame(self.preproc_frame)
        self.make_info_button(temp_frame, "left", 'train')
        self.split_lb = tkinter.Label(temp_frame,text="% Train Data: ",font=med_font,background="white",padx=5)
        self.split_lb.pack(side="left")
        self.add_dropdown([60,65,70,75,80,85,90],temp_frame,self.tr_te_split,wid=6)

        # create checkboxes for centering and scaling
        self.check_center_var = tkinter.IntVar()
        self.check_center_var.set(0)
        self.check_scale_var = tkinter.IntVar()
        self.check_scale_var.set(0)

        temp_frame = self.make_temp_frame(self.preproc_frame)
        self.make_info_button(temp_frame, "left", 'center')
        self.cb_center = tkinter.Checkbutton(temp_frame,text="Center",variable=self.check_center_var,
                                             font=med_font,background="white",padx=5,pady=5)
        self.cb_center.pack()

        temp_frame = self.make_temp_frame(self.preproc_frame)
        self.make_info_button(temp_frame,"left",'scale')
        self.cb_scale = tkinter.Checkbutton(temp_frame,text="Scale", variable=self.check_scale_var,
                                            font=med_font, background="white",padx=5,pady=5)
        self.cb_scale.pack()

        #  create frames for model, params, and submit button
        self.model_frame = tkinter.Frame(self.main_window)
        self.model_frame.pack(pady=10,padx=10)
        self.model_frame.config(background="white")
        self.param_frame = tkinter.Frame(self.main_window)
        self.param_frame.config(background="white")
        self.param_frame.pack(pady=10,padx=10)
        self.inner_frame = tkinter.Frame(self.param_frame)
        self.inner_frame.config(background="white")
        self.inner_frame.pack(pady=5,padx=5)
        self.bottom_frame = tkinter.Frame(self.main_window)
        self.bottom_frame.pack(padx=10)
        self.bottom_frame.config(background="white")

        self.modeling_lb = tkinter.Label(self.model_frame, text = "Model Building",font=big_font,background="white")
        self.modeling_lb.pack()
        self.model_label = tkinter.Label(self.model_frame, text="Choose a model",font=med_font,background="white")
        self.model_label.pack()

        # call helper function to add dropdown of different model options
        temp_frame = self.make_temp_frame(self.model_frame)
        self.make_info_button(temp_frame,"left",'model')
        self.add_dropdown(all_model_options, temp_frame, self.model_change,padx=10)

        # Create submit button
        self.submit_button = tkinter.Button(self.bottom_frame, text="Submit",command=self.get_model,font=big_font)
        self.submit_button.pack()
        tkinter.mainloop()

    def check_user_input(self):
        #check that a model is selected
        if self.X_train is None:
            tkMessageBox.showerror("InputError", "Please select a train/test split!")
            return False
        if self.model is None:
            tkMessageBox.showerror("InputError", "Please select a model!")
            return False
        #check that there are the correct number of parameters
        if len(self.param_list)==0:
            curr_bool=False
            tkMessageBox.showerror("InputError", "Please select parameters for the model!")
            return False
        #check if all the params with values are the same as the expected parameters
        if False in [self.param_list[i]in self.params.keys() for i in range(len(self.param_list))]:
            return False
        return True

    def get_model(self):
        if self.check_user_input():
            train_data = self.X_train
            test_data = self.X_test
            if self.check_center_var.get() and self.check_scale_var.get():
                sc = StandardScaler().fit(train_data)
                train_data = sc.transform(train_data)
                test_data = sc.transform(test_data)
            elif self.check_center_var.get():
                sc = StandardScaler(with_std=False).fit(train_data)
                train_data = sc.transform(train_data)
                test_data = sc.transform(test_data)
            elif self.check_scale_var.get():
                sc = StandardScaler(with_mean=False).fit(train_data)
                train_data = sc.transform(train_data)
                test_data = sc.transform(test_data)

            self.X_train_proc = train_data
            self.X_test_proc = test_data

            param_names = self.model._get_param_names()
            if 'random_state' in param_names:
                self.params['random_state'] = self.seed
            model = self.model(**self.params)
            self.modelfit = model.fit(train_data,self.y_train)
            self.predicted = self.modelfit.predict(test_data)
            self.display_data()
        else:
            if len(self.param_list) > 0:
                tkMessageBox.showerror("InputError", "Please select parameters for the model!")



    def display_data(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.tight_layout(pad=4)
        axes[0,0].set_title("Confusion Matrix")
        confusion_matrix = metrics.confusion_matrix(self.y_test,self.predicted)
        metrics.ConfusionMatrixDisplay(confusion_matrix).plot(ax=axes[0,0],colorbar=False)


        # add text annotation with accuracy sc  ore and params
        accuracy = metrics.accuracy_score(self.y_test, self.predicted)
        precision = metrics.precision_score(self.y_test,self.predicted,average='weighted')
        recall = metrics.recall_score(self.y_test,self.predicted,average='weighted')
        f1 = metrics.f1_score(self.y_test, self.predicted,average='weighted')
        scores = [accuracy,precision,recall,f1]
        score_names = ["Accuracy", "Precision", "Recall", "F1"]
        axes[0,1].annotate(f"{self.model.__name__}",(0,0.9),size=12,xycoords='axes fraction',va='center')
        param_key_list = list(self.params.keys())
        curr_height = 0.8
        for i in range(len(param_key_list)):
            key = param_key_list[i]
            axes[0,1].annotate(f"{key}: {self.params[key]}",(0,curr_height),size=13,xycoords='axes fraction',va='center')
            curr_height = curr_height - 0.1
        for i in range(len(scores)):
            axes[0,1].annotate(f"{score_names[i]}: {round(scores[i]*100,2)} ",(0,curr_height-0.15-0.1*i),size=14,xycoords='axes fraction',va='center')
        axes[0,1].axis('off')


        # plot decision boundary on train data
        axes[1, 0].scatter(self.X_train_proc[:, 0], self.X_train_proc[:, 1], c=self.y_train, cmap='viridis',edgecolor="k")
        axes[1, 0].set_xlabel("Predictor 1")
        axes[1, 0].set_ylabel("Predictor 2")
        axes[1,0].set_title("Decision Boundary and Training Data")

        DecisionBoundaryDisplay.from_estimator(self.modelfit,
                                               self.X_train_proc,
                                               ax=axes[1,0],
                                               response_method="predict",
                                               alpha=0.3,
                                               plot_method="contourf")


        corr_idx = [i for i in range(len(self.predicted)) if self.predicted[i] == self.y_test[i]]
        not_corr_idx = [i for i in range(len(self.predicted)) if self.predicted[i]!=self.y_test[i]]
        axes[1,1].scatter(self.X_test_proc[corr_idx,0],self.X_test_proc[corr_idx,1],c=self.y_test[corr_idx], cmap='viridis',edgecolor='k')
        axes[1,1].scatter(self.X_test_proc[not_corr_idx,0],self.X_test_proc[not_corr_idx,1],c=self.y_test[not_corr_idx],cmap='viridis',edgecolor='r')
        DecisionBoundaryDisplay.from_estimator(self.modelfit,
                                               self.X_train_proc,
                                               ax=axes[1, 1],
                                               response_method="predict",
                                               alpha=0.3,
                                               plot_method="contourf")
        axes[1, 1].set_xlabel("Predictor 1")
        axes[1, 1].set_ylabel("Predictor 2")
        axes[1, 1].set_title("Decision Boundary and Test Data")
        plt.show()







data = []

gui = myGUI()