from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
import pandas as pd
import pickle
import csv
import os

# Declare a Flask app
app = Flask(__name__)

path = os.path.dirname(__file__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # If a form is submitted
    if request.method == "POST":

        clf = pickle.load(open('model.pkl', 'rb'))

        f = request.files['report']
        f.save(secure_filename(f.filename))

        data = []
        with open(f.filename) as file:
            reader = csv.DictReader(file)

            [data.append(dict(row)) for row in reader]

        input_df = pd.DataFrame(data)
        input_df['Combined Diagnosis'] = input_df['Diagnosis'] + input_df['Gross Description'] + input_df[
            'Microscopic Description']
        input_df = input_df[['Combined Diagnosis']]

        # Get prediction
        predictions, output = clf.predict(input_df['Combined Diagnosis'].values.tolist())
        prediction = pd.DataFrame(predictions, columns=['Cancerous?'])
        prediction = pd.concat([input_df, prediction], axis=1)
        prediction[["accepted-rejected", "comments"]] = ""
        print(prediction.head())
        print(len(prediction))

        pred_csv = prediction.to_csv(os.path.join(path, r"preds.csv"))
        return redirect(url_for('reportgradingwebpage'))

    else:
        prediction = ""
        return render_template("index.html")


dataset_location = os.path.join(path, r"preds.csv")


@app.route('/reportgradingwebpage', methods=['GET', 'POST'])
def reportgradingwebpage():
    # variable to hold CSV data
    data = []
    # read data from CSV file

    with open(dataset_location) as f:
        # create CSV dictionary reader instance
        reader = csv.DictReader(f)

        # init CSV dataset
        [data.append(dict(row)) for row in reader]
        print(data)

        # print(data)
        row_number = 0  # initialise row number to zero
        number_of_reports = len(data)  # for the total number of reports in the html

        if request.method == "POST":
            if request.values.get("report-number-input"):
                row_number = int(request.values.get("report-number-input"))

            if request.values.get("accept-button"):
                # print(request.values.get("accept-button")) to see output values
                row_number = int(
                    request.values.get("accept-button"))  # stored row number in the button value so can access it

                # Read csv into dataframe
                df = pd.read_csv(dataset_location)
                # print(df) to debug
                # print(row_number) # to check if row number is updated

                # edit cell based on cell value row, column
                # https://re-thought.com/how-to-change-or-update-a-cell-value-in-python-pandas-dataframe/
                df.iat[row_number, 3] = "Accepted"

                # write output
                df.to_csv(dataset_location, index=False)

                # to read the csv, repeated code
                # variable to hold CSV data
                data = []

                # read data from CSV file

                with open(dataset_location) as f:
                    # create CSV dictionary reader instance
                    reader = csv.DictReader(f)

                    # init CSV dataset
                    [data.append(dict(row)) for row in reader]

                    # print(data)
                    number_of_reports = len(data)  # for the total number of reports in the html

            elif request.values.get("reject-button"):
                # print(request.values.get("reject-button")) to see output values
                row_number = int(
                    request.values.get("reject-button"))  # stored row number in the button value so can access it

                # Read csv into dataframe
                df = pd.read_csv(dataset_location)
                # print(df) to debug
                # print(row_number) # to check if row number is updated

                # edit cell based on cell value row, column
                # https://re-thought.com/how-to-change-or-update-a-cell-value-in-python-pandas-dataframe/
                df.iat[row_number, 3] = "Rejected"

                # write output
                df.to_csv(dataset_location, index=False)

                # to read the csv, repeated code
                # variable to hold CSV data
                data = []

                # read data from CSV file

                with open(dataset_location) as f:
                    # create CSV dictionary reader instance
                    reader = csv.DictReader(f)

                    # init CSV dataset
                    [data.append(dict(row)) for row in reader]

                    # print(data)
                    number_of_reports = len(data)  # for the total number of reports in the html

            elif request.values.get("comments-given-input"):
                # print(request.values.get("comments-given-input")) # to see output values
                comment = request.values.get("comments-given-input")

                row_number = int(request.values.get(
                    "comment-submit-button"))  # stored row number in the button value so can access it

                # Read csv into dataframe
                df = pd.read_csv(dataset_location)
                # print(df) # to debug
                # print(comment) # to check if row number is updated
                # print(row_number) # to check if row number is updated

                df.iat[row_number, 4] = comment

                # write output
                df.to_csv(dataset_location, index=False)

                # to read the csv, repeated code, could be stored in a function
                # variable to hold CSV data
                data = []

                # read data from CSV file

                with open(dataset_location) as f:
                    # create CSV dictionary reader instance
                    reader = csv.DictReader(f)

                    # init CSV dataset
                    [data.append(dict(row)) for row in reader]

                    # print(data)
                    number_of_reports = len(data)  # for the total number of reports in the html

        if row_number >= number_of_reports or row_number < 0:
            row_number = 0

        # print(row_number) console print for debugging

    # render HTML page dynamically
    return render_template("reportgradingwebpage.html", data=data, list=list, len=len, str=str, row_number=row_number,
                           number_of_reports=number_of_reports)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
