FROM python:3

run pip3 install fastapi uvicorn 

COPY . /turnover_model

CMD ['uvicorn', 'app.turnover_model:app', '--host', '0.0.0.0', '--port', '15400']
