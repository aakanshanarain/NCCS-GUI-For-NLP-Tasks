from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
import pandas as pd
import pickle
import csv
import os
import rule_based_model as rule_based_model

# Declare a Flask app
app = Flask(__name__)

path = os.path.dirname(__file__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # If a form is submitted
    if request.method == "POST":
        req = ""
        for filename, file in request.files.items():
            req = request.files[filename].name
        print(req)
        if req == "model1":
            clf = pickle.load(open('model.pkl', 'rb'))

            f = request.files[req]
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
            return redirect(url_for('cancerprediction'))

        elif req == "model2":
            clf = pickle.load(open('model2.pkl', 'rb'))

            f = request.files[req]
            f.save(secure_filename(f.filename))

            data = []
            with open(f.filename) as file:
                reader = csv.DictReader(file)

                [data.append(dict(row)) for row in reader]

            input_df = pd.DataFrame(data)
            input_df['Combined Diagnosis'] = input_df['Diagnosis'] + input_df['Gross Description'] + input_df[
                'Microscopic Description']
            input_df = input_df[['Combined Diagnosis']]
            columns = ('AMPULLA', 'ANAL CANAL', 'ANORECTAL JUNCTION', 'ANTERIOR SEGMENT', 'ANUS', 'APPENDIX', 'ASCENDING)', 'AXILLA (LEFT)', 'BLADDER', 'BREAST', 'BREAST (LEFT)', 'BREAST (RIGHT)', 'CAECUM', 'CAECUM/ILEOCAECAL JUNCTION', 'CERVIX', 'CHEST WALL', 'CHEST WALL (LEFT)', 'COLON', 'COLON (ANASTOMOTIC SITE)', 'COLON (ASCENDING)', 'COLON (CAECUM', 'COLON (CAECUM)', 'COLON (DESCENDING)', 'COLON (DISTAL TRANSVERSE)', 'COLON (HEPATIC FLEXURE)', 'COLON (PROXIMAL SIGMOID)', 'COLON (RECTOSIGMOID)', 'COLON (RECTUM)', 'COLON (RIGHT)', 'COLON (SIGMOID)', 'COLON (SPLENIC FLEXURE)', 'COLON (TRANSVERSE)', 'COLON (UPPER RECTUM)', 'COLORECTAL', 'COLORECTAL (PRIMARY)', 'DUODENUM', 'ENDOMETRIUM', 'ESOPHAGUS', 'FALLOPIAN TUBE (LEFT)', 'FALLOPIAN TUBE (RIGHT)', 'FOOT', 'GASTRIC', 'HEPATIC FLEXURE', 'KIDNEY', 'KIDNEY (LEFT)', 'LARGE BOWEL', 'LIVER', 'LIVER (LEFT LOBE)', 'LIVER (SEGMENT II)', 'LIVER SEGMENT 7/8 NODULE', 'LIVER SEGMENT 8 NODULE', 'LIVER SEGMENT II', 'LUNG', 'LUNG (LEFT LOWER LOBE)', 'LUNG (RIGHT LOWER LOBE)', 'LUNG (RIGHT UPPER LOBE)', 'LUNG (RIGHT)', 'MANDIBLE', 'NASOPHARYNX', 'OVARY', 'PARASTERNAL (LEFT)', 'PAROTID (LEFT)', 'PERITONEUM', 'PROSTATE', 'RECTAL', 'RECTOSIGMOID', 'RECTUM', 'RENAL (LEFT)', 'RENAL (RIGHT)', 'RIGHT LOWER LOBE', 'RIGHT UPPER LOBE', 'SALIVARY GLAND', 'SIGMOID', 'SPLENIC FLEXURE', 'THIGH', 'THYMUS', 'TONGUE', 'TONSIL', 'UNKNOWN', 'UTERUS')
            # Get prediction
            predictions, output = clf.predict(input_df['Combined Diagnosis'].values.tolist())
            prediction = pd.DataFrame(predictions,
                                      columns=['AMPULLA', 'ANAL CANAL', 'ANORECTAL JUNCTION', 'ANTERIOR SEGMENT', 'ANUS', 'APPENDIX', 'ASCENDING)', 'AXILLA (LEFT)', 'BLADDER', 'BREAST', 'BREAST (LEFT)', 'BREAST (RIGHT)', 'CAECUM', 'CAECUM/ILEOCAECAL JUNCTION', 'CERVIX', 'CHEST WALL', 'CHEST WALL (LEFT)', 'COLON', 'COLON (ANASTOMOTIC SITE)', 'COLON (ASCENDING)', 'COLON (CAECUM', 'COLON (CAECUM)', 'COLON (DESCENDING)', 'COLON (DISTAL TRANSVERSE)', 'COLON (HEPATIC FLEXURE)', 'COLON (PROXIMAL SIGMOID)', 'COLON (RECTOSIGMOID)', 'COLON (RECTUM)', 'COLON (RIGHT)', 'COLON (SIGMOID)', 'COLON (SPLENIC FLEXURE)', 'COLON (TRANSVERSE)', 'COLON (UPPER RECTUM)', 'COLORECTAL', 'COLORECTAL (PRIMARY)', 'DUODENUM', 'ENDOMETRIUM', 'ESOPHAGUS', 'FALLOPIAN TUBE (LEFT)', 'FALLOPIAN TUBE (RIGHT)', 'FOOT', 'GASTRIC', 'HEPATIC FLEXURE', 'KIDNEY', 'KIDNEY (LEFT)', 'LARGE BOWEL', 'LIVER', 'LIVER (LEFT LOBE)', 'LIVER (SEGMENT II)', 'LIVER SEGMENT 7/8 NODULE', 'LIVER SEGMENT 8 NODULE', 'LIVER SEGMENT II', 'LUNG', 'LUNG (LEFT LOWER LOBE)', 'LUNG (RIGHT LOWER LOBE)', 'LUNG (RIGHT UPPER LOBE)', 'LUNG (RIGHT)', 'MANDIBLE', 'NASOPHARYNX', 'OVARY', 'PARASTERNAL (LEFT)', 'PAROTID (LEFT)', 'PERITONEUM', 'PROSTATE', 'RECTAL', 'RECTOSIGMOID', 'RECTUM', 'RENAL (LEFT)', 'RENAL (RIGHT)', 'RIGHT LOWER LOBE', 'RIGHT UPPER LOBE', 'SALIVARY GLAND', 'SIGMOID', 'SPLENIC FLEXURE', 'THIGH', 'THYMUS', 'TONGUE', 'TONSIL', 'UNKNOWN', 'UTERUS'])
            prediction = pd.concat([input_df, prediction], axis=1)
            prediction[["Predicted Primary Site(s)"]] = ""
            for index, row in prediction.iterrows():
                sites = ""
                for organ in columns:
                    if row[organ] == 1:
                        sites += ", " + str(organ)
                if sites == "":
                    sites = "None Predicted"
                prediction.iat[index, 81] = sites

            prediction[["accepted-rejected", "comments"]] = ""
            print(prediction.head())
            print(len(prediction))

            pred_csv = prediction.to_csv(os.path.join(path, r"preds.csv"))
            return redirect(url_for('primarysitedetection'))

        elif req == "model3":
            # get the file uploaded
            f = request.files[req]
            f.save(secure_filename(f.filename))

            # link to regex model
            # from rule_based_model code
            df_initial = pd.read_csv(f.filename)
            df_initial.columns = [x.upper() for x in df_initial.columns]
            df_initial = df_initial[["SCM GUIDE", "DIAGNOSIS", "MICROSCOPIC DESCRIPTION",
                                     "GRADE(1, 2, 3, MILDLY OR WELL = 1, MODERATELY = 2, POORLY = 3)"]]
            print(df_initial)
            df_initial.rename(columns={'GRADE(1, 2, 3, MILDLY OR WELL = 1, MODERATELY = 2, POORLY = 3)': 'grades'},
                              inplace=True)
            print(df_initial)
            df_initial['grades'] = df_initial['grades'].fillna(
                0)  # this changes all NaN values in the grade column to 0
            print(df_initial)

            # Step 1: Convert DF to show ID, TEXT and GRADES
            df = rule_based_model.convert_df(df_initial)

            # Step 2: Find text matches to the word 'grade' and 'differentiated' and store in list (+ MATCHES)
            df = rule_based_model.find_matches(df)

            # Step 3: Determine the list of grades from the list of matches (+ DETERMINED)
            df = rule_based_model.determine_grade(df)

            # Step 4: Evaulate if determined grade is "Correct" or "Wrong" and calculate overall accuracy score (+ RESULT)
            df, accuracy_score = rule_based_model.evaluate_accuracy(df)

            # create the new columns
            df[["accepted-rejected", "comments"]] = ""

            pred_csv = df.to_csv(os.path.join(path, r'preds.csv'))
            return redirect(url_for('rulebasedmodelwebpage'))

        else:
            pass

    else:
        prediction = ""
        return render_template("index.html")


dataset_location = os.path.join(path, r"preds.csv")


@app.route('/cancerprediction', methods=['GET', 'POST'])
def cancerprediction():
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
    return render_template("cancerpredictionmodel.html", data=data, list=list, len=len, str=str, row_number=row_number,
                           number_of_reports=number_of_reports)


@app.route('/primarysitedetection', methods=['GET', 'POST'])
def primarysitedetection():
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
                df.iat[row_number, 83] = "Accepted"

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
                df.iat[row_number, 83] = "Rejected"

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
                df.iat[row_number, 84] = comment

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
    return render_template("primarysitepredictionmodel.html", data=data, list=list, len=len, str=str, row_number=row_number,
                           number_of_reports=number_of_reports)


@app.route('/rulebasedmodelwebpage', methods=["POST", "GET"])
def rulebasedmodelwebpage():
    # variable to hold CSV data
    data = []

    # read data from CSV file

    with open(dataset_location) as f:
        # create CSV dictionary reader instance
        reader = csv.DictReader(f)

        # init CSV dataset
        [data.append(dict(row)) for row in reader]

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
                df.iat[row_number, 7] = "Accepted"

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
                df.iat[row_number, 7] = "Rejected"

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

                df.iat[row_number, 8] = comment

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
    return render_template("cancergradepredictionmodel.html", data=data, list=list, len=len, str=str, row_number=row_number,
                           number_of_reports=number_of_reports)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
