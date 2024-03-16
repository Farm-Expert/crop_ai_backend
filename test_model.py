# import argparse
# import joblib
# import sklearn

# parser=argparse.ArgumentParser()

# parser.add_argument("-N",help="N",type=float)
# parser.add_argument("-P",help="P",type=float)
# parser.add_argument("-K",help="K",type=float)
# parser.add_argument("-temperature",help="temperature",type=float)
# parser.add_argument("-humidity",help="humidity",type=float)
# parser.add_argument("-ph",help="ph",type=float)
# parser.add_argument("-rainfall",help="rainfall",type=float)
# parser.add_argument("-label",help="label",type=float)

# arg=parser.parse_args

# def predict():
#     model=joblib.load('Farm_expert_model.pickle')
#     data_in1=[[arg.N,arg.P,arg.K,arg.temperature,arg.humidity,arg.ph,arg.rainfall]]
#     data_in2=[[arg.N,arg.P,arg.K,arg.temperature,arg.humidity,arg.ph,arg.rainfall,arg.label]]
#     prediction=model.predict(data_in1,data_in2)

#     # prediction1=model2.predict()
#     prob=model.predict_proba([data_in1,data_in2]).max()
#     print(prediction,prob)

# predict()
