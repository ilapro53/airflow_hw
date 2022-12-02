import os
import sys
import dill
import json
import re

import pandas as pd

from datetime import datetime
from pydantic import BaseModel
from sklearn.pipeline import Pipeline




try:
    path = os.environ['AIRFLOW_HOME']
    os.environ['PROJECT_PATH'] = path
    sys.path.insert(0, path)

except KeyError as e:
    if e.args[0] == 'AIRFLOW_HOME':
        path = 'C:\\airflow_hw'
        # Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
        os.environ['PROJECT_PATH'] = path
        # Добавим путь к коду проекта в $PATH, чтобы импортировать функции
        sys.path.insert(0, path)

    else:
        raise e



class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int



def predict():

    def load_model() -> Pipeline:
        models_path = os.path.join(path, 'data', 'models')
        models_files = os.listdir(models_path)
        pattern = re.compile(r'.*.pkl')

        model = None
        for file in models_files:
            if re.fullmatch(pattern, file) is not None:
                model = file

            break

        if model is None:
            raise FileNotFoundError(f'no .pkl files in "{path}"')

        if isinstance(model, str):
            with open(os.path.join(models_path, model), 'rb') as f:
                model = dill.load(f)

        else:
            raise FileNotFoundError(f'.pkl file named "{model}" not found in "{path}"')

        return model


    def load_tests_data():
        tests_path = os.path.join(path, 'data', 'test')
        tests_files = os.listdir(tests_path)
        pattern = re.compile(r'.*.json')

        tests = []
        load_exceptions = []
        validation_exceptions = []

        for file in tests_files:
            if re.fullmatch(pattern, file) is not None:

                try:
                    with open(os.path.join(tests_path, file), 'r') as f:
                        j = json.load(f)

                except Exception as e:
                    load_exceptions.append({
                        'file_name': file,
                        'path': os.path.join(tests_path, file),
                        'exception_object': e
                    })
                    continue


                try:
                    Form.parse_obj(j)

                except Exception as e:
                    validation_exceptions.append({
                            'file_name': file,
                            'path': os.path.join(tests_path, file),
                            'object': j,
                            'exception_object': e
                        })
                    continue

                tests.append({
                            'file_name': file,
                            'path': os.path.join(tests_path, file),
                            'object': j,
                        })

        return dict(
            tests = tests,
            load_exceptions = load_exceptions,
            validation_exceptions = validation_exceptions
        )


    model = load_model()
    tests, load_exceptions, validation_exceptions = load_tests_data().values()

    result = pd.DataFrame(columns=['id', 'prediction'])

    for test in tests:
        df = pd.DataFrame.from_dict([test['object']])
        y = model.predict(df)

        test = pd.DataFrame({
            'id': test['object']['id'],
            'prediction': y[0]
        }, index=[0])

        result = pd.concat([result, test], ignore_index=True)

    csv_filename = f'prediction_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    result.to_csv(os.path.join(path, 'data', 'predictions', csv_filename))

if __name__ == '__main__':
    predict()
