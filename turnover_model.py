#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# turnover_model.py

#These following packages are necessary to make the API work

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from fastapi import File, UploadFile

from fastapi.responses import JSONResponse

import pickle

import pandas as pd

import io

import numpy as np

from tabla_to import To_table

 

# We iniziate our app by FastAPI:

app = FastAPI()

 

# We load our model to make predictions:

xgb_model = pickle.load(open('TO_XGB_Definitivo.sav', 'rb'))

 

# Let's create a operation decorator:

 

@app.post("/upload/")

async def upload_csv(file: UploadFile = File()):

    #Let's use try to stop the process if an error happens

    try:

        # Ensure that the uploaded file is a CSV

        if file.content_type != "text/csv":

            return JSONResponse(content={"error": "Only CSV files are allowed"}, status_code=400)

       

        # Let's read our file and convert it into a dataframe with pandas:

        to_model_df = pd.read_csv(io.BytesIO(await file.read()))

       

        #These following lines are necessary to process our data:

       

        # We receive a dataframe but we only need some columns from it:

        variables = to_model_df[['time_position', 'time_to_increase', 'life', 'age', 'time_role', 'work_location_city',

                                 'cost_type2']].copy()

       

        #When we have CAPEX in the column cost_type2 we need to change that value to Other:

        variables['cost_type2'] = np.where(variables.cost_type2=='CAPEX', 'Other', variables.cost_type2)

       

        #These are the cities our model knows, then we need to take only the necessary ones:

        cities = ['AGUASCALIENTES', 'SANTO DOMINGO', 'CIUDAD DE MEXICO', 'BOGOTA D.C',

              'JALISCO','ZACATECAS','ESTADO DE MEXICO']

       

        # If the city is not in our list cities, then we change it to Other

        variables['work_location_city'] = np.where(variables['work_location_city'].isin(cities), variables['work_location_city'], 'Other')

       

        # We call our ML model to make predictions

        y_pred = xgb_model.predict(variables)

 

        # We create the variable prediction_of_probability to get the probabilities:

        prediction_of_probability = xgb_model.predict_proba(variables)

       

        # In the following three lines we add the predictions and probabilities to our dataframe as new columns

        variables['prediccion'] = y_pred

        variables['prob_0'] = prediction_of_probability[:,0]

        variables['prob_1'] = prediction_of_probability[:,1]

       

        # We store the results in a table (mi_tabla_challenge)

        To_table(variables)

       

        # WE convert out dataframe to a JSON

        results_data = variables.to_dict(orient='records')

       

        # We return the results as a downloadable file

        response = JSONResponse(content={"results": results_data})

        response.headers["Content-Disposition"] = "attachment; filename=results.csv"

        return response

       

        # The exception arises if there is a issue with the file:

    except Exception as e:

        return JSONResponse(content={"error": f"Error processing the file: {str(e)}"}, status_code=500)

