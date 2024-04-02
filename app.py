# https://youtu.be/bluclMxiUkA
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.

"""

import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
#model1 = pickle.load(open('models/rfc_model.pkl', 'rb'))pip show flask
model2 = pickle.load(open('models/svm_model.pkl', 'rb'))
model3 = pickle.load(open('models/ls_model.pkl', 'rb'))
model4 = pickle.load(open('models/gb_model.pkl', 'rb'))
model5 = pickle.load(open('models/xgb_model.pkl', 'rb'))

def vote(votes):
    result = {}  # สร้างพจนานุกรมเพื่อเก็บผลลัพธ์
    
    # นับโหวตแต่ละตัวเลือก
    for choice in votes:
        if choice in result:
            result[choice] += 1
        else:
            result[choice] = 1
    
    return result  # คืนค่าผลลัพธ์ที่ได้

def get_vote(votes):
    
    result = vote(votes)  # เรียกใช้ฟังก์ชัน vote เพื่อนับโหวต
    print(result)

    #หาตัวเลือกที่มีโหวตมากที่สุด
    winner = max(result, key=result.get)
    #print(winner)

    max_votes = max(result.values())
   # print(max_votes)

    total_votes = sum(result.values())  # นับรวมจำนวนโหวตทั้งหมด
    #print(total_votes)

    # หาค่าที่โหวตเยอะที่สุดและคำนวณเปอร์เซ็นต์การโหวต
    winner_percentage = round((max_votes / total_votes) * 100,2)
    #print(winner_percentage)

    return winner,winner_percentage  # คืนค่าตัวเลือกที่มีโหวตมากที่สุด


#import joblib
#model = joblib.load('models/model.pkl')

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():

    
    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    print(int_features) #int_features = [30.0,4500.0,0.0,90.0,30.0,10.0,10.0,10.0,25.0,20.0]

    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    print(features) #features = [[30.0,4500.0,0.0,90.0,30.0,10.0,10.0,10.0,25.0,20.0]]
    
    #prediction1 = model1.predict(features)  # features Must be in the form [[a, b]]
    #print(prediction1[0])

    prediction2 = model2.predict(features)  # features Must be in the form [[a, b]]
    print(prediction2[0])

    prediction3 = model3.predict(features)  # features Must be in the form [[a, b]]
    print(prediction3[0])

    prediction4 = model4.predict(features)  # features Must be in the form [[a, b]]
    print(prediction4[0])

    prediction5 = model5.predict(features)  # features Must be in the form [[a, b]]
    print(prediction5[0])

    #output,percent_output = get_vote([prediction1[0],prediction2[0],prediction3[0],prediction4[0],prediction5[0]])
    
    output,percent_output = get_vote([prediction2[0],prediction3[0],prediction4[0],prediction5[0]])
    
    print("output:", output)
    print("percent_output:", percent_output)

    if output == 0:
        result = "Poor"
    elif output == 1:
        result = "Standard"
    elif output == 2:
        result = "Good"
    else: 
        result = ""

    #output = round(prediction[0], 2)
    #output = "GOOD"
    return render_template('index.html', prediction_text='Your Credit Score : {}'.format(result))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()