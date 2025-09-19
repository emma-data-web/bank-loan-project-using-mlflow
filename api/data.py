import requests

url = "http://127.0.0.1:5000/predict"

data ={
  'purpose': "debt_consolidation",
  'int.rate': 0.2, 
  'installment':1.4,
  'log.annual.inc':2.4,
  'dti':1,
  'fico':3,
  'days.with.cr.line':0.81,
  'revol.bal':1.4, 
  'revol.util':1,
  'inq.last.6mths':0.2, 
  'delinq.2yrs':0, 
  'pub.rec':0, 
  'not.fully.paid': 0

}

response = requests.post(url, json=data)

print(response.text)