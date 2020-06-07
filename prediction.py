import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
#messagebox.showinfo("hello world")

root=tk.Tk()
path = "pro2.png"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(root, image = img)

#The Pack geometry manager packs widgets in rows or columns.
panel.pack(side = "bottom", fill = "both", expand = "yes")

OPTIONS_Graph=["Histogram","Bar Graph","Scatter Plot","DistPlot","BoxPlot","HeatMap","Decision Tree"]

OPTIONS= ["K-Means Classifier",
"Decision Tree Classifier","Decision Tree Classifier(Depth=3)",
"Random Forest Classifier","Random Forest Classifier(n_estimators=150)",
"Gradient Boosting Classifier","Gradient Boosting Classifier(Depth=1)","Gradient Boosting Classifier(Learning_rate=0.01)",
"SV Classifier","SV Classifier(C=1000)","SV Classifier(Random_state=42)",
"Logistics Regression","Logistics Regression(C=150)","Logistics Regression(C=0.01)",
"Min Max Scalar"]

root.geometry('1500x1250')
root.config(bg="lightblue1")
diabetes=pd.read_csv('dataset.csv')
#diabetes=pd.read_csv('CSV/dataset.csv')
label_title = tk.Label(root,borderwidth = 3,relief="sunken",text="   DIABETES PREDICTION   ",font=('Times New Roman', 25,'bold'),bg='black',fg="white")
label_title.place(x=460,y=39)
'''
background_image=tk.PhotoImage(file="/home/dell/Desktop/log_coef100.png")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
'''
X_train, X_test, Y_train, Y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=None)
diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8] 
training_accuracy = []
test_accuracy = []

# try n_neighbors from 1 to 21
neighbors_settings = range(1, 21)


def clear_text():
     #entry1.delete(0, 'end')
     txt=str("")
     label_clear = tk.Label(root, text=txt,height=210,width=400,font=('arial', 12))
     label_clear.place(x=300,y=180,width=700,height=400)
     

def dimensions():
    
    clear_text()
    ans1=str(format(diabetes.shape))
    ans2=str(diabetes.groupby('Outcome').size())
    label1 = tk.Label(root, text="(Rows,Columns)\n"+ans1+"\n",font=('arial', 12))
    label2 = tk.Label(root, text="Outcome is\n"+ans2,font=('arial', 12))
    label1.place(x=300,y=180,width=300,height=400)
    label2.place(x=700,y=180,width=400,height=400)
    

def columns():
	clear_text()
	col=[] #print("\nColumns for diabetes dataset: \n")
	for x in diabetes:
		col.append(x)
	name_col=str(col)
	label1 = tk.Label(root, text=name_col,font=('arial', 12))
	label1.place(x=300,y=180,width=700,height=400)
	
	
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
def callback(*args):
    clear_text()
    algorithm=str()
    algorithm=str(variable.get())
    if(algorithm=="K-Means Classifier"):
         knn = KNeighborsClassifier(n_neighbors=9)
         knn.fit(X_train, Y_train)
         acc_train=str(round(knn.score(X_train, Y_train)*100,5))
         acc_test=str(round(knn.score(X_test, Y_test)*100,5))
         label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+" %",font=("arial",13,"bold"))
         label1.place(x=300,y=180,width=700,height=400)
         for n_neighbors in neighbors_settings:
             # build the model
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, Y_train)
            # record training set accuracy
            training_accuracy.append(knn.score(X_train, Y_train))
            # record test set accuracy
            test_accuracy.append(knn.score(X_test, Y_test))
         plt.title("K-Means")
         #plt.figure(figsize=(8,8))
         plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
         #plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
         plt.ylabel("Accuracy")
         plt.xlabel("n_neighbors")
         plt.legend()
         plt.show()
         plt.savefig('knn_compare_model')
     
    elif(algorithm=="Decision Tree Classifier"):
        
        tree = DecisionTreeClassifier()
        tree.fit(X_train, Y_train)
        acc_train=str(round(tree.score(X_train, Y_train)*100,5))
        acc_test=str(round(tree.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(tree)
        plt.savefig('feature_importance')
        plt.show()
        #plt.set_position(self, bottom, which='both')    
        
    elif(algorithm=="Random Forest Classifier"):
        
        rf = RandomForestClassifier()
        rf.fit(X_train, Y_train)
        acc_train=str(round(rf.score(X_train, Y_train)*100,5))
        acc_test=str(round(rf.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(rf)
        plt.savefig('feature_importance_rf')
        plt.show()
        
    elif(algorithm=="Gradient Boosting Classifier"):
        #clear_text()
        gb = GradientBoostingClassifier()
        gb.fit(X_train, Y_train)
        acc_train=str(round(gb.score(X_train, Y_train)*100,5))
        acc_test=str(round(gb.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(gb)
        plt.savefig('feature_importance_gb')
        plt.show()
        
    elif(algorithm=="SV Classifier"):
        
        svc = SVC()
        svc.fit(X_train, Y_train)
        acc_train=str(round(svc.score(X_train, Y_train)*100,5))
        acc_test=str(round(svc.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(svc)
        plt.savefig('feature_importance_svc')
        plt.show()
    
    elif(algorithm=="Logistics Regression"):
        
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        acc_train=str(round(logreg.score(X_train, Y_train)*100,5))
        acc_test=str(round(logreg.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
    
        plt.figure(figsize=(8,6))	
        plt.plot(logreg.coef_.T, 'o', label="C=1")
        plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
        plt.hlines(0, 0, diabetes.shape[1])
        plt.ylim(-5, 5)
        plt.xlabel("Feature")
        plt.ylabel("Coefficient magnitude")
        plt.legend()
        plt.savefig('log_coef')
        plt.show()
        
    elif(algorithm=="Logistics Regression(C=150)"):   
        
        logreg100 = LogisticRegression(C=150).fit(X_train, Y_train)
        acc_train=str(round(logreg100.score(X_train, Y_train)*100,5))
        acc_test=str(round(logreg100.score(X_test, Y_test)*100,5))
        
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
    
        plt.figure(figsize=(8,6))	
        plt.plot(logreg100.coef_.T, 'o', label="C=1")
        plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
        plt.hlines(0, 0, diabetes.shape[1])
        plt.ylim(-5, 5)
        plt.xlabel("Feature")
        plt.ylabel("Coefficient magnitude")
        plt.legend()
        plt.savefig('log_coef100')
        plt.show()
        

    elif(algorithm=="Logistics Regression(C=0.01)"):
        
        logreg001 = LogisticRegression(C=0.01).fit(X_train, Y_train)
        acc_train=str(round(logreg001.score(X_train, Y_train)*100,5))
        acc_test=str(round(logreg001.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
    
        plt.figure(figsize=(8,6))	
        plt.plot(logreg001.coef_.T, 'o', label="C=1")
        plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
        plt.hlines(0, 0, diabetes.shape[1])
        plt.ylim(-5, 5)
        plt.xlabel("Feature")
        plt.ylabel("Coefficient magnitude")
        plt.legend()
        plt.savefig('log_coef001')
        plt.show()


    elif(algorithm=="Decision Tree Classifier(Depth=3)"):
        \
        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X_train, Y_train)
        acc_train=str(round(tree.score(X_train, Y_train)*100,5)) 
        acc_test=str(round(tree.score(X_test, Y_test)*100,5))
        
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(tree)
        plt.savefig('feature_importance')
        plt.show()

    elif(algorithm=="Random Forest Classifier(n_estimators=150)"):
        
        rf = RandomForestClassifier(n_estimators=150, random_state=0)
        rf.fit(X_train, Y_train)
        acc_train=str(round(rf.score(X_train, Y_train)*100,5))
        acc_test=str(round(rf.score(X_test, Y_test)*100,5))
        
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(rf)
        plt.savefig('feature_importance_rf')
        plt.show()
    
    elif(algorithm=="Gradient Boosting Classifier(Depth=1)"):
        
        gb1 = GradientBoostingClassifier(random_state=0, max_depth=1)
        gb1.fit(X_train, Y_train)
        acc_train=str(round(gb1.score(X_train, Y_train)*100,5))
        acc_test=str(round(gb1.score(X_test, Y_test)*100,5))
        
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(gb1)
        plt.savefig('feature_importance_gb1')
        plt.show()
        
    elif(algorithm=="Gradient Boosting Classifier(Learning_rate=0.01)"):    
        
        gb2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
        gb2.fit(X_train, Y_train)
        acc_train=str(round(gb2.score(X_train, Y_train)*100,5))
        acc_test=str(round(gb2.score(X_test, Y_test)*100,5))
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(gb2)
        plt.savefig('feature_importance_gb2')
        plt.show()
            
    elif(algorithm=="SV Classifier(Random_state=42)"):
        
        svc=SVC(kernel='linear',random_state=42)
        svc.fit(X_train, Y_train)
        acc_train=str(round(svc.score(X_train, Y_train)*100,5))
        acc_test=str(round(svc.score(X_test, Y_test)*100,5))
    
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(svc)
        plt.savefig('feature_importance_svc(42)')
        plt.show()
    
    elif(algorithm=="Min Max Scalar"):
        
        scaler = MinMaxScaler()
        svc = SVC()
        
        X_train_scaled = scaler.fit_transform(X_train)*100
        X_test_scaled = scaler.fit_transform(X_test)*100
        svc.fit(X_train_scaled, Y_train)
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+str(round(X_train_scaled,5))+"%\n\nTest Set: "+str(round(X_test_scaled,5))+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(svc)
        plt.savefig('feature_importance_svc(42)')
        plt.show()
        

    elif(algorithm=="SV Classifier(C=1000)"):
        
        svc = SVC(C=1000)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        svc.fit(X_train_scaled, Y_train)
        acc_train=str(round(svc.score(X_train, Y_train)*100,5))
        acc_test=str(round(svc.score(X_test, Y_test)*100,5))
        
        label1 = tk.Label(root, text=algorithm+"\n\nTraining Set: "+acc_train+"%\n\nTest Set: "+acc_test+"%",font=("arial",13,"bold"))
        label1.place(x=300,y=180,width=700,height=400)
        
        plot_feature_importances_diabetes(svc)
        plt.savefig('feature_importance_svc(42)')
        plt.show()

def chart(*args):
	
	clear_text()
	graph=str()
	graph=str(variable1.get())
	if(graph=="Bar Graph"):
		fig = plt.figure()
		fig.suptitle('Bar Graph for Outcome', fontsize=20)
		sns.countplot(diabetes['Outcome'])
		plt.show()
		
		
	elif(graph=="Histogram"):
		#fig=plt.figure(figsize=(20,3))
		#fig.suptitle('Histogram', fontsize=20)
		diabetes.hist(figsize=(20,3))
		plt.show()
		
	elif(graph=="DistPlot"):
		fig, ax2 = plt.subplots(4, 2, figsize=(13, 13))
		sns.distplot(diabetes['Pregnancies'],ax=ax2[0][0])
		sns.distplot(diabetes['Plasma'],ax=ax2[0][1])
		sns.distplot(diabetes['Pressure'],ax=ax2[1][0])
		sns.distplot(diabetes['Skin Thickness'],ax=ax2[1][1])
		sns.distplot(diabetes['Insulin'],ax=ax2[2][0])
		sns.distplot(diabetes['BMI'],ax=ax2[2][1])
		sns.distplot(diabetes['DiaPedigree'],ax=ax2[3][0])
		sns.distplot(diabetes['Age'],ax=ax2[3][1])
		plt.show()
	
	elif(graph=="BoxPlot"):
		fig=plt.figure(figsize=(20,3))
		for i in np.arange(1,7):
	    		data3=plt.subplot(1,7,i,title=diabetes.columns[i])
	    		sns.boxplot(diabetes[diabetes.columns[i]])
		plt.show()
	
	elif(graph=="Scatter Plot"):
		fig=plt.figure(figsize=(20,3))
		sns.pairplot(diabetes,hue='Outcome')
		plt.show()
		
		
	elif(graph=="HeatMap"):
		fig=plt.figure(figsize=(20,3))
		cor=diabetes.corr()
		sns.heatmap(cor,annot=True)
		plt.show()
	
	elif(graph=="Decision Tree"):
		fig=plt.figure(figsize=(20,3))
		tree = DecisionTreeClassifier(max_depth=3, random_state=0)
		tree.fit(X_train, Y_train)
		img=mpimg.imread('dtree2.png')
		
		imgplot = plt.imshow(img)
		plt.show()
		#plt.show(export_graphviz(tree,out_file="diabetes_census_tree.dot",class_names="insu",feature_names=diabetes.columns,impurity=False,filled=True))
		

		
def cmatrix():
	clear_text()
	svc=SVC(kernel='linear',random_state=42)
	svc.fit(X_train, Y_train)
	#print("SVC Prediction\n\n")
	y_pred=svc.predict(X_test)
	label1 = tk.Label(root, text="\n\nConfusion Matrix :\n"+str(confusion_matrix(Y_test,y_pred)),font=("arial",13,"bold"))
	label1.place(x=550,y=280,width=150,height=200)
	svc=SVC(kernel='linear',random_state=42)
	svc.fit(X_train, Y_train)
	
	cm = np.array(confusion_matrix(Y_test,y_pred))
	true_pos = np.diag(cm)
	false_pos = np.sum(cm, axis=0) - true_pos
	false_neg = np.sum(cm, axis=1) - true_pos
	precision = np.sum(true_pos / (true_pos + false_pos))
	recall = np.sum(true_pos / (true_pos + false_neg))
	label2 = tk.Label(root, text="\n\nPrecision : "+str(round(precision,7)),font=('arial', 13,'bold'))
	label2.place(x=400,y=200,width=190,height=100)
	label3 = tk.Label(root, text="\n\nRecall : "+str(round(recall,7)),font=('arial', 13,'bold'))
	label3.place(x=600,y=200,width=360,height=100)
	
button1 = tk.Button (root, text='Dimensions',relief=tk.RIDGE,width=15,command=dimensions,font=('Arial', 13, 'bold'),bg='white',fg="black",activebackground="#274193",activeforeground='black') 
button1.place(x=40,y=200,width=200,height=45)

button2 = tk.Button (root, text='Columns' ,relief=tk.SUNKEN,command=columns,font=('Arial', 13, 'bold'),bg='white',fg="black",activebackground="#274193",activeforeground='black')
button2.place(x=40,y=250,width=200,height=45)

variable = tk.StringVar(root)
#chooseTest=tk.OptionMenu(self, self.selectedOption, *optionsList)
#chooseTest.config(font=helv35)
variable.set("Check Accuracy") # default value
w = tk.OptionMenu(root, variable, *OPTIONS)
w.config(font='bold')
w.config(font='Arial')
w.place(x=40,y=300,width=200,height=45)
w.config(bg = "white") 

labelTest = tk.Label(text="", font=('Helvetica', 13))
labelTest.place(x=410,y=200,height=45)
variable.trace("w", callback)

button3=tk.Button(root,text="Clear",command=clear_text,relief=tk.RAISED,font=("Arial",13,'bold'),bg='white',fg="black",activebackground="#274193",activeforeground='black')
#button3.place(x=40,y=350,width=200,height=45)
button3.place(x=40,y=500,width=200,height=45)

variable1 = tk.StringVar(root)
variable1.set("Graph") # default value
w = tk.OptionMenu(root, variable1, *OPTIONS_Graph)
w.config(font='bold')
w.config(font='Arial')
w.config(font='13')
w.place(x=40,y=400,width=200,height=45)
w.config(bg = "white")

labelTest = tk.Label(text="", font=('Helvetica', 13))
labelTest.place(x=410,y=200,height=45)
variable1.trace("w", chart)  


button5=tk.Button(root,text="Exit",command=root.destroy,relief=tk.GROOVE,font=("Arial",13,'bold'),bg='white',fg="black",activebackground="#274193",activeforeground='black')
#button5.place(x=40,y=450,width=200,height=45)
button5.place(x=40,y=550,width=200,height=45)


button6=tk.Button(root,text="Confusion Matrix",command=cmatrix,relief=tk.GROOVE,font=("Arial",13,'bold'),bg='white',fg="black",activebackground="#274193",activeforeground='black')
#button6.place(x=40,y=550,width=200,height=45)
button6.place(x=40,y=450,width=200,height=45)

def dele():
	e1_entry.delete(0,END)
	e2_entry.delete(0,END)
	e3_entry.delete(0,END)
	e4_entry.delete(0,END)
	e5_entry.delete(0,END)
	e6_entry.delete(0,END)
	e7_entry.delete(0,END)
	e8_entry.delete(0,END)
	
	

def createNewWindow():
	top=tk.Toplevel(root)
	top.geometry("400x500")
	top.configure(bg='grey33')
	global e1,e2,e3,e4,e5,e6,e7,e8
	global e1_entry,e2_entry,e3_entry,e4_entry,e5_entry,e6_entry,e7_entry,e8_entry
	e1=tk.StringVar()
	e1_entry=tk.Entry(top,textvariable=e1).place(x = 200, y = 50)
	e2=tk.StringVar()
	e3=tk.StringVar()
	e4=tk.StringVar()
	e5=tk.StringVar()
	e6=tk.StringVar()
	e7=tk.StringVar()
	e8=tk.StringVar()
	e2_entry=tk.Entry(top,textvariable=e2).place(x = 200, y= 90)
	e3_entry=tk.Entry(top,textvariable=e3).place(x = 200, y = 130)
	e4_entry=tk.Entry(top,textvariable=e4).place(x = 200, y = 170)
	e5_entry=tk.Entry(top,textvariable=e5).place(x = 200, y = 210)
	e6_entry=tk.Entry(top,textvariable=e6).place(x = 200, y = 250)
	e7_entry=tk.Entry(top,textvariable=e7).place(x = 200, y = 290)
	e8_entry=tk.Entry(top,textvariable=e8).place(x = 200, y = 330)
	global a1
	a1=e1.get()
	global a2
	a2=e2.get()
	global a3
	a3=e3.get()
	global a4
	a4=e4.get()
	global a5
	a5=e5.get()
	global a6
	a6=e6.get()
	global a7
	a7=e7.get()
	global a8
	a8=e8.get()
	Pregnancies = tk.Label(top, text = "Pregnancies").place(x = 30,y = 50,width= 130,height= 25)
	Glucose = tk.Label(top, text = "Glucose").place(x = 30,y = 90,width= 130,height= 25)
	BloodPressure= tk.Label(top, text = "BloodPressure").place(x = 30,y = 130,width= 130,height= 25)
	SkinThickness= tk.Label(top, text = "SkinThickness").place(x = 30,y = 170,width= 130,height= 25)
	Insulin = tk.Label(top, text = "Insulin").place(x = 30,y = 210,width= 130,height= 25)
	BMI =tk. Label(top, text = "BMI").place(x = 30,y = 250,width= 130,height= 25)
	DiabetesPedigree = tk.Label(top, text = "DiabetesPedigree").place(x = 30, y = 290,width= 130,height= 25)
	Age= tk.Label(top, text = "Age").place(x = 30, y = 330,width= 130,height= 25)
	sbmitbtn = tk.Button(top, text = "Predict",activebackground = "pink", activeforeground = "blue",command=predict_func).place(x = 150, y = 390)
	
	resetbtn = tk.Button(top, text = "Reset",activebackground = "pink", activeforeground = "blue", command=dele).place(x =250, y = 390)
	
	top.mainloop()
	
def predict_func():
	
	a1=float(e1.get())
	a2=float(e2.get())
	a3=float(e3.get())
	a4=float(e4.get())
	a5=float(e5.get())
	a6=float(e6.get())
	a7=float(e7.get())
	a8=float(e8.get())
	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.fit_transform(X_test)
	scaler.fit(X_train)
	svc=SVC(kernel='linear',random_state=42)
	svc.fit(X_train, Y_train)
	new_df=pd.DataFrame([[a1,a2,a3,a4,a5,a6,a7,a8]])
	new_df_scaled=scaler.transform(new_df)
	prediction=svc.predict(new_df)
	svc.fit(X_train_scaled, Y_train)
	if prediction==1:
		messagebox.showinfo("Result","You are prone to diabetes with accuracy "+str(round(svc.score(X_train_scaled, Y_train),6)))
		#print("\nDIABETES TEST POSITIVE\n")	round(precision,7)
	else:
		messagebox.showinfo("Result","You are not prone to diabetes with accuracy "+str(round(svc.score(X_train_scaled, Y_train),6)))
		#print("\nDIABETES TEST NEGATIVE\n")

	
#app = tk.Tk()

#button=tk.Button(root,text="Prediction",font("Arial",13,'bold'),bg='#5CC7B2',fg="black",activebackground="#274193",activeforeground='black',command=createNewWindow).place(x=150,y=450,width=200,height=45)

button = tk.Button(root, text='Prediction' ,command=createNewWindow,font=('Arial', 13, 'bold'),bg='white',fg="black",activebackground="#274193",activeforeground='black')
#button.place(x=40,y=500,width=200,height=45)
button.place(x=40,y=350,width=200,height=45)
#button = tk.Button(root,text="Prediction",activebackground="#274193",command=createNewWindow).place(x=150,y=450,width=200,height=45)
#stackoverflow
#sbmitbtn = tk.Button(win, text='Open', command=load)
#sbmitbtn.pack(expand=tk.FALSE, fill=tk.X, side=tk.TOP)
	

root.mainloop()
