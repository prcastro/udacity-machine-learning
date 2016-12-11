#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

number_people = len(enron_data)
print("Number of people in the dataset: %d"%number_people)

number_features = len(enron_data.values()[0])
print("Features for each person: %d"%number_features)

number_of_pois = sum([person['poi'] for person in enron_data.values()])
print("Number of POIs in the dataset: %d"%number_of_pois)

prentice_stocks = enron_data['PRENTICE JAMES']['total_stock_value']
print("Total stock value of James Prentice: %d"%prentice_stocks)

colwell_poi_emails = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print("Number of email sent to POI from Wesley Colwell: %d"%colwell_poi_emails)

skilling_stock_options = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print("Number of stock options exercised by Jeffrey K Skilling: %d"%skilling_stock_options)

available_salaries = sum([person['salary'] != 'NaN' for person in enron_data.values()])
print("Number of salaries available: %d"%available_salaries)

available_emails = sum([person['email_address'] != 'NaN' for person in enron_data.values()])
print("Number of emails available: %d"%available_emails)
