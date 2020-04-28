from flask import Flask, request, render_template
from joblib import load
import numpy as np
import pandas as pd
import ast
import json


app = Flask(__name__, static_url_path='')
# gears_module_calculator = load('./ai/gears_module_calculator.joblib')
# normal_gears_module_calculator = load('./ai/normal_gears_module_calculator.joblib')
# dMin_ejes = load('./ai/dMin_ejes.joblib')
# Lmin_chavetas = load('./ai/Lmin_chavetas.joblib')
# ny_chavetas = load('./ai/ny_chavetas.joblib')
# ny_reten = load('./ai/ny_reten.joblib')

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/gears/helical', methods=['POST'])
def helicoidal_gears_module():
	# content = request.get_data().decode("UTF-8")
	# content = ast.literal_eval(content)

	# gear = [content["body"]["teeth"],content["body"]["mateial"],content["body"]["rev"],content["body"]["Mt"],content["body"]["degree"],content["body"]["precision"],content["body"]["security"]]	
	# module = gears_module_calculator.predict(pd.DataFrame([gear]))
	# result = json.dumps({"data": round(module[0],2)})
	return json.dumps({"data": 3})

@app.route('/gears', methods=['POST'])
def gears_module():
  # content = request.get_data().decode("UTF-8")
  # content = ast.literal_eval(content)

  # gear = [content["body"]["teeth"],content["body"]["mateial"],content["body"]["rev"],content["body"]["Mt"],content["body"]["precision"],content["body"]["security"]]  
  # module = normal_gears_module_calculator.predict(pd.DataFrame([gear]))
  # result = json.dumps({"data": round(module[0],2)})
  return json.dumps({"data": 3.5})

@app.route('/axis', methods=['POST'])
def axis():
  # content = request.get_data().decode("UTF-8")
  # content = ast.literal_eval(content)

  # axi = [content["body"]["sut"],content["body"]["bhm"],content["body"]["Tm"],content["body"]["Mt"],content["body"]["ny"]]  
  # module = dMin_ejes.predict(pd.DataFrame([axi]))
  # result = json.dumps({"data": round(module[0],2)})
  return json.dumps({"data": 100})

@app.route('/key/lmin', methods=['POST'])
def key_lmin():
  # content = request.get_data().decode("UTF-8")
  # content = ast.literal_eval(content)

  # key = [content["body"]["axi_dim"],content["body"]["sy"],content["body"]["ny"],content["body"]["Tm"]]  
  # module = Lmin_chavetas.predict(pd.DataFrame([key]))
  # result = json.dumps({"data": round(module[0],2)})
  return json.dumps({"data": 2})

@app.route('/key/ny', methods=['POST'])
def key_ny():
  # content = request.get_data().decode("UTF-8")
  # content = ast.literal_eval(content)

  # key = [content["body"]["axi_dim"],content["body"]["sut"],content["body"]["bhm"],content["body"]["Mt"],content["body"]["Tm"]]  
  # module = ny_chavetas.predict(pd.DataFrame([key]))
  # result = json.dumps({"data": round(module[0],2)})
  return json.dumps({"data": 1.5})

@app.route('/picket', methods=['POST'])
def picket():
  # content = request.get_data().decode("UTF-8")
  # content = ast.literal_eval(content)

  # picket = [content["body"]["axi_dim"],content["body"]["sut"],content["body"]["bhm"],content["body"]["Mt"],content["body"]["Tm"]]  
  # module = ny_reten.predict(pd.DataFrame([picket]))
  # result = json.dumps({"data": round(module[0],2)})
  return json.dumps({"data": 2})

@app.route('/forces', methods=['POST'])
def forces():
  content = request.get_data().decode("UTF-8")
  content = ast.literal_eval(content)
  A = np.array([content["body"]["force"],content["body"]["momentum"]])
  xInt = np.size(A,0)
  yInt = np.size(A,1)

  At = A.transpose()

  C = At.dot(A)

  esx = np.size(C,0)
  wai = np.size(C,1)
  D = C[:esx-1,:wai-1]
  E = C[esx-1,:wai-1]
  F = np.linalg.solve(D,E)

  return json.dumps({"data":list(F)})

if __name__ == "__main__":
	app.run()